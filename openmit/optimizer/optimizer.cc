#include "openmit/optimizer/optimizer.h"
//#include "openmit/optimizer/adadelta.h"
//#include "openmit/optimizer/adagrad.h"
//#include "openmit/optimizer/adam.h"
//#include "openmit/optimizer/ftrl.h"
//#include "openmit/optimizer/rmsprop.h"
#include "openmit/optimizer/sgd.h"
#include "openmit/optimizer/als.h"
namespace mit {

Optimizer * Optimizer::Create(const mit::KWArgs & kwargs, 
                              const std::string & name) {
  std::string optimizer;
  if (name != "") {
    optimizer = name;
  } else {
    for (auto & kv : kwargs) {
      if (kv.first != "optimizer") continue;
      optimizer = kv.second;
    }
  }
  LOG(INFO) << "Optimizer optimizer: " << optimizer;
  if (optimizer == "gd" || optimizer == "sgd") {
    return mit::SGDOptimizer::Get(kwargs);
  } else if (optimizer == "adadelta") {
    return mit::SGDOptimizer::Get(kwargs);
    //return mit::AdaDeltaOptimizer::Create(kwargs);
  } else if (optimizer == "adagrad") {
    return mit::SGDOptimizer::Get(kwargs);
    //return mit::AdaGradOptimizer::Get(kwargs);
  } else if (optimizer == "adam") {
    return mit::SGDOptimizer::Get(kwargs);
    //return mit::AdamOptimizer::Get(kwargs);
  } else if (optimizer == "ftrl") {
    return mit::SGDOptimizer::Get(kwargs);
    //return mit::FtrlOptimizer::Get(kwargs);
  } else if (optimizer == "rmsprop") {
    return mit::SGDOptimizer::Get(kwargs);
    //return mit::RMSPropOptimizer::Get(kwargs);
  } else if (optimizer == "als") {
    return mit::ALSOptimizer::Get(kwargs);
  } 
  else {
    LOG(ERROR) << 
      "optimizer not in [gd, sgd, adagrad, rmsprop, adadelta, adam, " << 
      "ftrl lbfgs, als, ...]. " << 
      "optimizer: " << optimizer;
    return nullptr;
  }
} // Optimizer::Create

void Optimizer::Run(const ps::SArray<mit_uint> & keys, 
                    const ps::SArray<mit_float> & vals, 
                    const ps::SArray<int> & lens, 
                    std::unordered_map<mit_uint, mit::Entry *> * weight) {
  auto offset = 0u;
  auto keys_len = keys.size();
  for (auto i = 0u; i < keys_len; ++i) {
    ps::SArray<mit_float> entry_grad = vals.segment(offset, offset + lens[i]);
    // TODO OpenMP 
    for (auto idx = 0; idx < lens[i]; ++idx) {
      mit_float w = (*weight)[keys[i]]->Get(idx);
      mit_float g = entry_grad[idx];
      if (idx == 0) {       // update w
        Update(param_w_, keys[i], idx, g, w, (*weight)[keys[i]]);
      } else {              // update v
        Update(param_w_, keys[i], idx, g, w);
      }
      (*weight)[keys[i]]->Set(idx, w);
    }
    offset += lens[i];
  }
}

void Optimizer::Run(mit::SArray<mit_float> & grad, 
              mit::SArray<mit_float> * weight) {
  // TODO OpenMP?
  for (auto i = 0u; i < weight->size(); ++i) {
    Update(i, grad[i], (*weight)[i]);
  }
} // Optimizer::Run

} // namespace mit
