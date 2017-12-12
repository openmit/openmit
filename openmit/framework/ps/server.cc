#include "openmit/framework/ps/server.h"
#include "openmit/tools/monitor/transaction.h"
using namespace mit;

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
  
  // model for update 
  model_.reset(mit::Model::Create(kwargs));

  // parameter 
  complete_worker_number_ = 0;
}
  
Server::~Server() { 
  if (kv_server_) { delete kv_server_; kv_server_ = nullptr; }

  std::unordered_map<ps::Key, mit::Entry*>::iterator iter;
  iter = weight_.begin();
  while (iter != weight_.end()) {
    if (iter->second) {
      delete iter->second; iter->second = nullptr;
    }
    iter++;
  }
}

void Server::Run() { 
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this] { return exit_ == true; });
  LOG(INFO) << "role @server[" << ps::MyRank() << "] task finish.";
}

void Server::KVHandle(const ps::KVMeta & req_meta, 
                      const ps::KVPairs<mit_float> & req_data, 
                      ps::KVServer<mit_float> * server) {
  if (req_meta.push) {
    int cmd = req_meta.cmd;
    switch(cmd) {
      case signal::UPDATE: {
        if (cli_param_.debug) {
          LOG(INFO) << "@server[" << ps::MyRank() 
            << "] number of weight: " << weight_.size() 
            << ", grads: " << mit::DebugStr(req_data.vals.data(), 10);
        }
        model_->Update(
          req_data.keys, req_data.vals, req_data.lens, &weight_);
      } break;
      default:
        LOG(FATAL) << "cmd is not recoginized. " << req_meta.cmd;
    }
    server->Response(req_meta);
  } else { // pull
    ps::KVPairs<mit_float> response;
    PullRequest(req_data, response);
    server->Response(req_meta, response);
  }
}

void Server::CmdHandle(const ps::SimpleData & recved, ps::SimpleApp * app) {
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

void Server::PullRequest(const ps::KVPairs<mit_float> & req_data, ps::KVPairs<mit_float> & response) {
  response.keys = req_data.keys;
  response.vals.clear();
  //response.lens.clear();
  model_->Pull(response, &weight_);
}

void Server::ExitCondition() {
  mutex_.lock(); 
  complete_worker_number_++; 
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
  std::string dump_out = cli_param_.model_dump;
  std::string bin_out = cli_param_.model_binary;
  if (epoch == "") {
    dump_out += "/part-" + myrank;
    bin_out += "/last/part-" + myrank;
  } else {   // save middle result by epoch
    std::string postfix = "/iter-" + epoch + "/part-" + myrank;
    dump_out += ".middle" + postfix;
    bin_out += postfix;
  }
  std::unique_ptr<dmlc::Stream> dumpfo(
    dmlc::Stream::Create(dump_out.c_str(), "w"));
  SaveTextModel(dumpfo.get());
  std::unique_ptr<dmlc::Stream> binfo(
    dmlc::Stream::Create(bin_out.c_str(), "w"));
  SaveBinaryModel(binfo.get());
  LOG(INFO) << "@server[" + myrank + "] save model done.";
}


void Server::SaveTextModel(dmlc::Stream * fo) {
  std::unique_ptr<Transaction> trans(new Transaction(1, "server", "dump_out"));
  mit::EntryMeta * entry_meta = model_->EntryMeta();
  dmlc::ostream oss(fo);
  for (auto & kv : weight_) {
    auto key = kv.first;
    oss << key << "\t" << kv.second->String(entry_meta) << "\n";
  }
  // force flush before fo destruct 
  oss.set_stream(nullptr);
  Transaction::End(trans.get());
}

void Server::SaveBinaryModel(dmlc::Stream * fo) {
  std::unique_ptr<Transaction> trans(
    new Transaction(1, "server", "binary_out"));
  // save entry meta
  mit::EntryMeta * entry_meta = model_->EntryMeta();
  entry_meta->Save(fo);
  // save model 
  std::unordered_map<ps::Key, mit::Entry * >::iterator iter;
  iter = weight_.begin();
  while (iter != weight_.end()) {
    // TODO  key special process
    fo->Write((char *) &iter->first, sizeof(ps::Key));
    iter->second->Save(fo, entry_meta);
    iter++;
  }
  Transaction::End(trans.get());
} // method SaveBinaryModel

void Server::DumpModel(dmlc::Stream * fi, dmlc::Stream * fo) {
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
