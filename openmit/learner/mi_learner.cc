#include "openmit/framework/mpi/admm.h"
#include "openmit/framework/ps/ps.h"
#include "openmit/learner/mi_learner.h"

namespace mit {

MILearner * MILearner::Create(const mit::KWArgs & kwargs) {
  std::string framework = "ps";     // default
  for (auto & kv : kwargs) {
    if (kv.first != "framework") continue;
    framework = kv.second;
  }
  LOG(INFO) << "MILearner framework: " << framework;

  if (framework == "mpi") {
    return mit::Admm::Get(kwargs);
  } else if (framework == "ps") {
    return mit::PS::Get(kwargs);
  } else {
    LOG(ERROR) << "framework must be belonging to [mpi, ps]."
      << " current framework value: " << framework;
    return nullptr;
   }
}

} // namespace mit
