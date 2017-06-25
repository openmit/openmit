#include "openmit/optimizer/optimizer.h"
#include "openmit/optimizer/adagrad.h"
#include "openmit/optimizer/ftrl.h"
#include "openmit/optimizer/gd.h"

namespace mit {

Opt * Opt::Create(const mit::KWArgs & kwargs, std::string & optimizer) {
  if (optimizer == "gd" || optimizer == "gd") {
    return mit::GD::Get(kwargs);
  } else if (optimizer == "adagrad") {
    return mit::AdaGrad::Get(kwargs);
  } else if (optimizer == "ftrl") {
    return mit::Ftrl::Get(kwargs);
  } else {
    LOG(ERROR) << 
      "optimizer not in [gd, ftrl, lbfgs, als, ...]. " << 
      "aoptimizer: " << optimizer;
    return nullptr;
  }
}

} // namespace mit
