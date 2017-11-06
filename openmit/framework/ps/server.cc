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
  kv_server_->set_request_handle(
    std::bind(&Server::KVRequestHandle, this, 
              std::placeholders::_1, 
              std::placeholders::_2, 
              std::placeholders::_3));

  // model for update 
  model_.reset(mit::Model::Create(kwargs));
  model_->InitOptimizer(kwargs);

  // parameter 
  complete_worker_number_ = 0;
}
  
Server::~Server() {
  if (kv_server_) { delete kv_server_; }
}

void Server::
KVRequestHandle(const ps::KVMeta & req_meta, 
                const ps::KVPairs<mit_float> & req_data, 
                ps::KVServer<mit_float> * server) {
  if (req_meta.push) {
    int cmd = req_meta.cmd;
    switch(cmd) {
      case signal::UPDATE: Run(req_data); break;
      case signal::SAVEINFO:
        {
          std::string model_path = cli_param_.model_dump 
          + "/iter=" + std::to_string(req_data.keys[0]) 
            + "/part." + std::to_string(ps::MyRank());
          auto * fo = dmlc::Stream::Create(model_path.c_str(), "w"); 
          SaveModel(fo);
          delete fo;
        }
        break;
      case signal::FINISH: WorkerFinish(); break;
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

void Server::Run(const ps::KVPairs<mit_float> & req_data) { 
  if (cli_param_.debug) {
    LOG(INFO) << "@server[" << ps::MyRank() 
      << "] number of weight: " << weight_.size() 
      << ", grads: " << mit::DebugStr(req_data.vals.data(), 10);
  }
  model_->Update(
    req_data.keys, req_data.vals, req_data.lens, &weight_);
}

void Server::
PullRequest(const ps::KVPairs<mit_float> & req_data, 
            ps::KVPairs<mit_float> & response) {
  response.keys = req_data.keys;
  response.vals.clear();
  response.lens.clear();
  model_->Pull(response, &weight_);
}

void Server::WorkerFinish() {
  mutex_.lock(); 
  complete_worker_number_++; 
  mutex_.unlock();
  if (complete_worker_number_ == ps::NumWorkers()) {
    LOG(INFO) << "all workers completed. save model begin";
    std::string dump_out = cli_param_.model_dump + "/part-" + std::to_string(ps::MyRank());
    auto * dumpfo = dmlc::Stream::Create(dump_out.c_str(), "w");
    SaveModel(dumpfo); delete dumpfo;
    LOG(INFO) << "save model (text format) completed!";

    std::string binary_out = cli_param_.model_binary + "/last/part-" + std::to_string(ps::MyRank());
    auto * binfo = dmlc::Stream::Create(binary_out.c_str(), "w");
    SaveBinaryModel(binfo); delete binfo;
    LOG(INFO) << "save model (binary format) completed!";
    LOG(INFO) << "save model done.";
  }
}

void Server::SaveModel(dmlc::Stream * fo) {
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
  std::unique_ptr<Transaction> trans(new Transaction(1, "server", "binary_out"));
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
