#include "openmit/framework/ps/server.h"
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

  // model 
  model_.reset(mit::Model::Create(kwargs));
  model_->InitOptimizer(kwargs);

  // entry_meta_
  mit::ModelParam model_param;
  model_param.InitAllowUnknown(kwargs);
  entry_meta_.reset(new mit::EntryMeta(model_param));
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
          SaveModel(foo);
          // model binary
          // TODO
        }
        break;
      default:
        LOG(FATAL) 
          << "Server Handle. cmd is not recoginized. " 
          << req_meta.cmd;
    }
    server->Response(req_meta);
  } else { // pull
    ps::KVPairs<mit_float> response;
    ProcessPullRequest(req_data, response);
    server->Response(req_meta, response);
  }
}

void Server::
ProcessPullRequest(const ps::KVPairs<mit_float> & req_data, 
                   ps::KVPairs<mit_float> & response) {
  response.keys = req_data.keys;
  response.vals.clear();
  response.lens.clear();
  model_->Pull(response, entry_meta_.get(), &weight_);
  if (cli_param_.debug) {
    LOG(INFO) << "pull request vals info: " 
      << mit::DebugStr(response.vals.data(), 10); 
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

void Server::SaveModel(dmlc::Stream * fo) {
  dmlc::ostream oss(fo);
  oss << "server save model\n";
  for (auto & kv : weight_) {
    auto key = kv.first;
    mit::Entry * entry = kv.second;
    oss << key << "\t" << entry->Size() << "\n";
  }
}

void Server::DumpModel(dmlc::Stream * fi, dmlc::Stream * fo) {
  dmlc::ostream oss(fo);
  // TODO
}
} // namespace mit
