#include "openmit/optimizer/optimizer.h"
//#include "openmit/optimizer/adadelta.h"
//#include "openmit/optimizer/adagrad.h"
//#include "openmit/optimizer/adam.h"
//#include "openmit/optimizer/ftrl.h"
//#include "openmit/optimizer/rmsprop.h"
#include "openmit/optimizer/sgd.h"

namespace mit {

Optimizer * Optimizer::Create(const mit::KWArgs & kwargs, 
                  std::string & optimizer) {
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
  } else {
    LOG(ERROR) << 
      "optimizer not in [gd, sgd, adagrad, rmsprop, adadelta, adam, " << 
      "ftrl lbfgs, als, ...]. " << 
      "optimizer: " << optimizer;
    return nullptr;
  }
} // Optimizer::Create

void Optimizer::Run(const ps::SArray<mit_uint> & keys, const ps::SArray<mit_float> & vals, const ps::SArray<int> & lens, PMAPT1 * weight) {
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

void Optimizer::Run(PMAPT & grad, PMAPT * weight) {
  // OpenMP
  for (auto & kunit : grad) {
    auto key = kunit.first;
    mit::Unit * unit = kunit.second;
    auto size = unit->Size();
    CHECK(size >= 1) << "length of unit should not less than 1.";
    // OpenMP
    for (auto idx = 0u; idx < size; ++idx) {
      float w = (*weight)[key]->Get(idx);
      float g = grad[key]->Get(idx);
      // g += param_.l1 * 1 + param_.l2 * w;
      Update(key, idx, size, g, w);
      (*weight)[key]->Set(idx, w);
    }
  }
} // Optimizer::Run 

void Optimizer::Run(mit::SArray<mit_float> & grad, 
              mit::SArray<mit_float> * weight) {
  // TODO OpenMP?
  for (auto i = 0u; i < weight->size(); ++i) {
    Update(i, grad[i], (*weight)[i]);
  }
} // Optimizer::Run

} // namespace mit
