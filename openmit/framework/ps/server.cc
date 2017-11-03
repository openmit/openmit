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
      case signal::UPDATE: 
        { 
          Run(req_data); 
        }
        break;
      case signal::SAVEINFO:
        {
          std::string model_path = cli_param_.model_dump 
            + "/iter=" + std::to_string(req_data.keys[0]) 
            + "/part." + std::to_string(ps::MyRank());
          auto * fo = dmlc::Stream::Create(model_path.c_str(), "w"); 
          SaveModel(fo);
        }
        break;
      case signal::FINISH:
        {
          // model dump 
          std::string model_path = cli_param_.model_dump 
            + "/part." + std::to_string(ps::MyRank());
          auto * foo = dmlc::Stream::Create(model_path.c_str(), "w");
          //SaveModel(foo);
          WorkerFinish();
          // model binary
          // TODO
        }
        break;
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
  LOG(INFO) << "worker finish.";
  mutex_.lock(); 
  complete_worker_number_++; 
  mutex_.unlock();
  LOG(INFO) << "complete_worker_number_: " << complete_worker_number_ << ", ps::NumWorkers: " << ps::NumWorkers();
  if (complete_worker_number_ == ps::NumWorkers()) {
    LOG(INFO) << "   worker finish.";
    std::string dump_out = cli_param_.model_dump 
      + "/part-" + std::to_string(ps::MyRank());
    std::unique_ptr<dmlc::Stream> dumpfo(
      dmlc::Stream::Create(dump_out.c_str(), "w"));
    SaveModel(dumpfo.get());
    std::string binary_out = cli_param_.model_binary + 
      "/last/part-" + std::to_string(ps::MyRank());
    std::unique_ptr<dmlc::Stream> binfo(
      dmlc::Stream::Create(binary_out.c_str(), "w"));
    SaveBinaryModel(binfo.get());
    // BinaryModel
  }
}

void Server::SaveModel(dmlc::Stream * fo) {
  std::unique_ptr<Transaction> trans(
    new Transaction(1, "server", "dump_out"));
  dmlc::ostream oss(fo);
  for (auto & kv : weight_) {
    auto key = kv.first;
    oss << key << "\t" << kv.second->String() << "\n";
  }
  // force flush before fo destruct 
  oss.set_stream(nullptr);
  Transaction::End(trans.get());
}

void Server::SaveBinaryModel(dmlc::Stream * fo) {
  std::unique_ptr<Transaction> trans(
    new Transaction(1, "server", "binary_out"));
  dmlc::ostream oss(fo);
  for (auto & kv : weight_) {
    auto key = kv.first;
    oss << key << "\t" << kv.second->String() << "\n";
  }
  // force flush before fo destruct 
  oss.set_stream(nullptr);
  Transaction::End(trans.get());
} // SaveBinaryModel

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
