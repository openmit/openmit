#include "openmit/optimizer/optimizer.h"
#include "openmit/optimizer/adagrad.h"
#include "openmit/optimizer/ftrl.h"
#include "openmit/optimizer/sgd.h"

namespace mit {
Opt * Opt::Create(const mit::KWArgs & kwargs, 
                  std::string & optimizer) {
  if (optimizer == "gd" || optimizer == "sgd") {
    return mit::SGD::Get(kwargs);
  } else if (optimizer == "adagrad") {
    LOG(INFO) << "optimizer: " << optimizer;
    return mit::AdaGrad::Get(kwargs);
  } else if (optimizer == "ftrl") {
    LOG(INFO) << "optimizer: " << optimizer;
    return mit::Ftrl::Get(kwargs);
  } else {
    LOG(ERROR) << 
      "optimizer not in [gd, sgd, ftrl, adagrad, lbfgs, als, ...]. " << 
      "aoptimizer: " << optimizer;
    return nullptr;
  }
} // Opt::Create
} // namespace mit
