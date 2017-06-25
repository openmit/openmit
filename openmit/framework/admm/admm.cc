#include "openmit/framework/admm/admm.h"

namespace mit {

DMLC_REGISTER_PARAMETER(AdmmParam);

Admm::Admm(const mit::KWArgs & kwargs) {
  this->miparam_.InitAllowUnknown(kwargs);
  param_.InitAllowUnknown(kwargs);
  LOG(INFO) << "Admm param_.rho: " << param_.rho;
}

void Admm::Run() {
  // TODO
  LOG(INFO) << "Admm::Run()~";
}
} // namespace mit
