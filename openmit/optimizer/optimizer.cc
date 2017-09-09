#include "openmit/optimizer/optimizer.h"
#include "openmit/optimizer/adagrad.h"
#include "openmit/optimizer/adadelta.h"
#include "openmit/optimizer/ftrl.h"
#include "openmit/optimizer/rmsprop.h"
#include "openmit/optimizer/sgd.h"

namespace mit {
Opt * Opt::Create(const mit::KWArgs & kwargs, 
                  std::string & optimizer) {
  if (optimizer == "gd" || optimizer == "sgd") {
    return mit::SGD::Get(kwargs);
  } else if (optimizer == "adagrad") {
    return mit::AdaGrad::Get(kwargs);
  } else if (optimizer == "adadelta") {
    return mit::AdaDelta::Create(kwargs);
  } else if (optimizer == "ftrl") {
    return mit::Ftrl::Get(kwargs);
  } else if (optimizer == "rmsprop") {
    return mit::RMSProp::Get(kwargs);
  } else {
    LOG(ERROR) << 
      "optimizer not in [gd, sgd, ftrl, adagrad, lbfgs, als, ...]. " << 
      "aoptimizer: " << optimizer;
    return nullptr;
  }
} // Opt::Create

void Opt::Run(PMAPT & grad, PMAPT * weight) {
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
} // Opt::Run 
} // namespace mit
