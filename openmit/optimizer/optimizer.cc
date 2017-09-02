#include "openmit/optimizer/optimizer.h"
#include "openmit/optimizer/adagrad.h"
#include "openmit/optimizer/adadelta.h"
#include "openmit/optimizer/ftrl.h"
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
  } else {
    LOG(ERROR) << 
      "optimizer not in [gd, sgd, ftrl, adagrad, lbfgs, als, ...]. " << 
      "aoptimizer: " << optimizer;
    return nullptr;
  }
} // Opt::Create
} // namespace mit
