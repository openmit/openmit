#include <cstdlib>
#include "openmit/framework/server.h"

namespace mit {

Server::Server(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

void Server::Init(const mit::KWArgs & kwargs) {
  cli_param_.InitAllowUnknown(kwargs);

  // kv_server_
  kv_server_ = new ps::KVServer<mit_float>(0);
  using namespace std::placeholders;
  kv_server_->set_request_handle(
    std::bind(&Server::KVHandle, this, _1, _2, _3));

  // request & response process except kv logic_ 
  static_cast<ps::SimpleApp *>(kv_server_)->set_request_handle(
    std::bind(&Server::CmdHandle, this, _1, _2));
  static_cast<ps::SimpleApp *>(kv_server_)->set_response_handle(
    std::bind(&Server::CmdHandle, this, _1, _2));
  
  // model for update && pull
  model_.reset(mit::Model::Create(kwargs));

  // thread pool 
  //CHECK_GT(cli_param_.num_thread, 1);
  //thread_pool_.reset(new mit::ThreadPool(cli_param_.num_thread - 1));

  // parameter 
  complete_worker_number_ = 0;
}
  
Server::~Server() { 
  if (kv_server_) { delete kv_server_; kv_server_ = nullptr; }

  //std::unordered_map<ps::Key, mit::Entry*>::iterator iter;
  for (auto iter = weight_.begin(); iter != weight_.end(); iter++) {
    if (iter->second) {
      delete iter->second; iter->second = nullptr;
    }
  }
}

void Server::Run() { 
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this] { return exit_ == true; });
  LOG(INFO) << "number of key: " << weight_.size();
  LOG(INFO) << "role @server[" << ps::MyRank() << "] task finish.";
}

void Server::KVHandle(const ps::KVMeta& req_meta, 
                      const ps::KVPairs<mit_float>& req_data, 
                      ps::KVServer<mit_float>* server) {
  if (req_meta.push) {
    server->Response(req_meta);
    int cmd = req_meta.cmd;
    switch(cmd) {
      case signal::UPDATE: {
        if (cli_param_.debug) {
          std::string msg = "grads from worker " + mit::DebugStr(req_data.vals.data(), 5);
          LOG(INFO) << msg;
        }
        model_->Update(req_data.keys, req_data.vals, req_data.lens, &weight_);
        if (cli_param_.debug) LOG(INFO) << "update done";
      } break;
      default:
        LOG(FATAL) << "unknown cmd. " << req_meta.cmd;
    }
  } else { // pull 
    PullRequest(req_meta, req_data, server);
    /*
    thread_pool_->Append(
      [this, req_meta, req_data, server]() { 
        PullRequest(req_meta, req_data, server); 
      });
      */
  }
}

void Server::CmdHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
  ps::Message msg;
  msg.meta.head           = recved.head;
  msg.meta.body           = recved.body;
  msg.meta.timestamp      = recved.timestamp;
  msg.meta.request        = false;
  msg.meta.simple_app     = true;
  msg.meta.customer_id    = kv_server_->get_customer()->id();
  msg.meta.recver         = recved.sender;
  msg.meta.sender         = ps::Postoffice::Get()->van()->my_node().id;
  // msg ready, send it 
  ps::Postoffice::Get()->van()->Send(msg);

  int cmd = recved.head;
  switch(cmd) {
    case signal::WORKER_FINISH:
      {
        ExitCondition();
      }
      break;
    case signal::SAVE_EPOCH:
      {
        SaveModel(recved.body);  // epoch
      }
      break;
    default:
      LOG(FATAL) << "cmd is not recoginized. cmd: " << cmd;
  }
}

void Server::PullRequest(const ps::KVMeta& req_meta, 
                         const ps::KVPairs<mit_float>& req_data, 
                         ps::KVServer<mit_float>* server) {
  ps::KVPairs<mit_float> response;
  response.keys = req_data.keys;
  response.extras = req_data.extras;
  this->model_->Pull(response, &weight_);

  server->Response(req_meta, response);
}

void Server::ExitCondition() {
  mutex_.lock(); 
  complete_worker_number_++; 
  LOG(INFO) << "complete_worker_number_: " << complete_worker_number_;
  mutex_.unlock();
  if (complete_worker_number_ == ps::NumWorkers()) {
    SaveModel();
    std::string rank = std::to_string(ps::MyRank());
    kv_server_->Request(signal::SERVER_FINISH, rank, ps::kScheduler);
    mutex_.lock(); exit_ = true; mutex_.unlock();
    cond_.notify_all();
  }
}

void Server::SaveModel(std::string epoch) {
  std::string myrank = std::to_string(ps::MyRank());
  LOG(INFO) << "@server[" + myrank + "] save model begin";
  std::string dump_out = cli_param_.out_path + "/model_dump/last"; 
  std::string bin_out = cli_param_.out_path + "/model_binary";

  // check local file
  if (cli_param_.out_path.compare(0, 4, "hdfs") != 0  &&
      cli_param_.out_path.compare(0, 6, "viewfs") != 0 && 
      cli_param_.out_path.compare(0, 2, "s3") != 0) {
    LOG(INFO) << "local dir " << cli_param_.out_path;
    std::string cmd = "rm -rf " + dump_out + " || true"; 
    CHECK_EQ(system(cmd.c_str()), 0);
    cmd = "mkdir -p " + dump_out; 
    CHECK_EQ(system(cmd.c_str()), 0);
    cmd = "rm -rf " + bin_out + " || true";
    CHECK_EQ(system(cmd.c_str()), 0);
    cmd = "mkdir -p " + bin_out;
    CHECK_EQ(system(cmd.c_str()), 0);
  }

  dump_out += "/part-" + myrank;
  bin_out += "/part-" + myrank;
  
  std::unique_ptr<dmlc::Stream> dumpfo(dmlc::Stream::Create(dump_out.c_str(), "w"));
  SaveTextModel(dumpfo.get());
  std::unique_ptr<dmlc::Stream> binfo(dmlc::Stream::Create(bin_out.c_str(), "w"));
  SaveBinaryModel(binfo.get());
  LOG(INFO) << "@server[" + myrank + "] save model done.";
}


void Server::SaveTextModel(dmlc::Stream* fo) {
  mit::EntryMeta* entry_meta = model_->EntryMeta();
  dmlc::ostream oss(fo);
  if (cli_param_.model == "mf") {
    mit_uint index_threashold = ((mit_uint)1 << cli_param_.nbit);
    for (auto & kv : weight_) {
      auto key = kv.first;
      auto type = 0;
      if ((mit_uint)kv.first >= index_threashold) {
        key = DecodeField(key, cli_param_.nbit);
        type = 1;
      }
      oss << type << "\t" << key << "\t" << kv.second->String(entry_meta) << "\n";
    }
  }
  else {
    for (auto & kv : weight_) {
      auto key = kv.first;
      oss << key << "\t" << kv.second->String(entry_meta) << "\n";
    }
  }
  // force flush before fo destruct 
  oss.set_stream(nullptr);
}

void Server::SaveBinaryModel(dmlc::Stream* fo) {
  // save entry meta
  mit::EntryMeta* entry_meta = model_->EntryMeta();
  entry_meta->Save(fo);
  // save model 
  //std::unordered_map<ps::Key, mit::Entry* >::iterator iter;
  auto iter = weight_.begin();
  while (iter != weight_.end()) {
    // TODO  key special process
    fo->Write((char*) &iter->first, sizeof(ps::Key));
    iter->second->Save(fo, entry_meta);
    iter++;
  }
} // method SaveBinaryModel

void Server::DumpModel(dmlc::Stream* fi, dmlc::Stream* fo) {
  dmlc::ostream oss(fo);
  // TODO
  // force flush before fo destruct 
  oss.set_stream(nullptr);
}

void Server::LoadModel(dmlc::Stream * fi) {
  dmlc::istream iss(fi);
  // TODO 
  // force flush before fi destruct 
  iss.set_stream(nullptr);
}

} // namespace mit
