#include <functional>

#include "openmit/framework/ps/server.h"
using namespace mit;

namespace mit {

DMLC_REGISTER_PARAMETER(ServerParam);

Server::Server(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

void Server::Init(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);

  // kv_server_
  kv_server_ = new ps::KVServer<mit_float>(0);
  kv_server_->set_request_handle(std::bind(&Server::KVRequestHandle, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  // updater_
  updater_.reset(new mit::Updater(kwargs));
}
  
Server::~Server() {
  if (kv_server_) {
    std::cout << "~KVmitServer. " << std::endl;
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
        Run(&req_data);
        break;
      case signal::SAVEINFO:
        {
          std::string model_path = param_.model_out 
            + "/iter=" + std::to_string(req_data.keys[0]) 
            + "/part." + std::to_string(ps::MyRank());
          auto * fo = dmlc::Stream::Create(model_path.c_str(), "w"); 
          SaveModel(fo);
        }
        break;
      case signal::FINISH:
        {
          std::string model_path = param_.model_out 
            + "/part." + std::to_string(ps::MyRank());
          auto * foo = dmlc::Stream::Create(model_path.c_str(), "w");
          SaveModel(foo);
        }
        break;
      default:
        std::cout << "Server Handle. cmd is not recoginized. " << req_meta.cmd << std::endl;
    }
    server->Response(req_meta);
  } else { // pull
    ps::KVPairs<mit_float> response;
    response.keys = req_data.keys;
    response.vals.resize(req_data.keys.size());
    for (auto i = 0u; i < req_data.keys.size(); ++i) {
      ps::Key key = req_data.keys[i]; 
      if (weight_.find(key) == weight_.end()) {
        mit::Unit * unit = new Unit(param_.field_num * param_.k + 1);
        weight_.insert(std::make_pair(key, unit));
      }
      // Flating: unit -> vector
      response.vals[i] = weight_[req_data.keys[i]]->Get(0);
    }
    server->Response(req_meta, response);
  }
}

void Server::Run(const ps::KVPairs<mit_float> * req_data) { 
  updater_->Run(req_data, &weight_);
}

void Server::SaveModel(dmlc::Stream * fo) {
  dmlc::ostream oss(fo);
  oss << "server save model\n";
  for (auto & kunit : weight_) {
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
