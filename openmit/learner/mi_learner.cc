#include "openmit/framework/admm/admm.h"
#include "openmit/framework/ps/parameter_server.h"
#include "openmit/learner/mi_learner.h"

namespace mit {

DMLC_REGISTER_PARAMETER(MILearnerParam);

MILearner * MILearner::Create(const mit::KWArgs & kwargs) {
  MILearnerParam param_;
  param_.InitAllowUnknown(kwargs);
  if (param_.framework == "admm") {
    return mit::Admm::Get(kwargs);
  } else if (param_.framework == "ps") {
    return mit::PS::Get(kwargs);
  } else {
    LOG(ERROR) << "framework must be belonging to [admm, ps]."
      << " current framework value: " << param_.framework;
    return nullptr;
   }
}

} // namespace mit
