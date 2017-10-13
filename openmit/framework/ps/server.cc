#include "openmit/framework/ps/server.h"
using namespace mit;

namespace mit {

DMLC_REGISTER_PARAMETER(ServerParam);

Server::Server(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

void Server::Init(const mit::KWArgs & kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  param_.InitAllowUnknown(kwargs);

  // kv_server_
  kv_server_ = new ps::KVServer<mit_float>(0);
  kv_server_->set_request_handle(std::bind(&Server::KVRequestHandle, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  // optimizer 
  optimizer_.reset(mit::Optimizer::Create(kwargs));

  // entry_meta_
  entry_meta_.reset(new mit::EntryMeta(cli_param_));
}
  
Server::~Server() {
  if (kv_server_) {
    delete kv_server_;
  }
  // TODO
}

void Server::KVRequestHandle(const ps::KVMeta & req_meta, 
                             const ps::KVPairs<mit_float> & req_data, 
                             ps::KVServer<mit_float> * server) {
  if (req_meta.push) {
    int cmd = req_meta.cmd;
    switch(cmd) {
      case signal::UPDATE: 
        { 
          Run(req_data); 
          if (cli_param_.debug) {
            LOG(INFO) << "serverid: " << ps::MyRank() 
              << ", weight_.size: " << weight1_.size();
          }
        }
        break;
      case signal::SAVEINFO:
        {
          std::string model_path = param_.model_dump 
            + "/iter=" + std::to_string(req_data.keys[0]) 
            + "/part." + std::to_string(ps::MyRank());
          auto * fo = dmlc::Stream::Create(model_path.c_str(), "w"); 
          SaveModel(fo);
        }
        break;
      case signal::FINISH:
        {
          // model dump 
          std::string model_path = param_.model_dump 
            + "/part." + std::to_string(ps::MyRank());
          auto * foo = dmlc::Stream::Create(model_path.c_str(), "w");
          SaveModel(foo);
          // model binary
          // TODO
        }
        break;
      default:
        std::cout << "Server Handle. cmd is not recoginized. " << req_meta.cmd << std::endl;
    }
    server->Response(req_meta);
  } else { // pull
    ps::KVPairs<mit_float> response;
    ProcessPullRequest(req_data, response);
    server->Response(req_meta, response);
  }
}

void Server::ProcessPullRequest(const ps::KVPairs<mit_float> & req_data, ps::KVPairs<mit_float> & response) {
  response.keys = req_data.keys;
  response.vals.clear();
  response.lens.clear();
  if (cli_param_.data_format == "libfm") {
    for (auto i = 0u; i < response.keys.size(); ++i) {
      ps::Key key = response.keys[i];
      if (weight1_.find(key) == weight1_.end()) {
        size_t field_size = 0;
        mit_uint fieldid = 0l;
        if (key > 0l) {   // not bias item
          fieldid = mit::DecodeField(key, cli_param_.nbit);
          CHECK(fieldid > 0) << "fieldid error. fieldid: " << fieldid;
          field_size = entry_meta_->CombineInfo(fieldid)->size();
        }
        mit::Entry * entry = new mit::Entry(
          cli_param_, field_size, fieldid);
        weight1_.insert(std::make_pair(key, entry));
      }
      mit::Entry * entry = weight1_[key];
      ps::SArray<mit_float> wv; 
      wv.CopyFrom(entry->Data(), entry->length);
      // fill response.vals and response.lens
      response.vals.append(wv);
      response.lens.push_back(entry->length);
    }
  } else {  // data_format in ["auto", "libsvm"]
    for (auto i = 0u; i < response.keys.size(); ++i) {
      ps::Key key = response.keys[i];
      if (weight1_.find(key) == weight1_.end()) {
        mit::Entry * entry = new mit::Entry(cli_param_);
        weight1_.insert(std::make_pair(key, entry));
        if (cli_param_.debug) {
          LOG(INFO) << "key not in weight_, new it. " << key;
        }
      }
      mit::Entry * entry = weight1_[key];
      ps::SArray<mit_float> wv; 
      wv.CopyFrom(entry->Data(), entry->length);
      // fill response.vals and response.lens
      response.vals.append(wv);
      response.lens.push_back(entry->length);
    }
  }
  if (cli_param_.debug) {
    LOG(INFO) << "pull request vals info: " 
      << mit::DebugStr(response.vals.data(), 10); 
  }
}

void Server::Run(const ps::KVPairs<mit_float> & req_data) { 
  if (cli_param_.debug) {
    LOG(INFO) << "grads from worker: " 
      << mit::DebugStr(req_data.vals.data(), 10);
  }
  optimizer_->Run(req_data.keys, req_data.vals, req_data.lens, &weight1_);
}

void Server::SaveModel(dmlc::Stream * fo) {
  dmlc::ostream oss(fo);
  oss << "server save model\n";
  for (auto & kunit : weight1_) {
    auto feati = kunit.first;
    auto * unit = kunit.second;
    //if (unit->AllZero()) continue;
    oss << feati << "\t" << unit->Str() << "\n";
  }
}

void Server::DumpModel(dmlc::Stream * fi, dmlc::Stream * fo) {
  dmlc::ostream oss(fo);
  // TODO
}
} // namespace mit
