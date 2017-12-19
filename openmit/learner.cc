#include "openmit/framework/mpi/mpi_admm.h"
#include "openmit/framework/ps/ps.h"
#include "openmit/learner.h"

namespace mit {
MILearner * MILearner::Create(const mit::KWArgs& kwargs) {
  std::string framework = "ps";   // default 
  for (auto& kv : kwargs) {
    if (kv.first != "framework") continue;
    framework = kv.second;
  }
  if (framework == "mpi") {
    return mit::MPIAdmm::Get(kwargs);
  } else if (framework == "ps") {
    return mit::PS::Get(kwargs);
  } else {
    LOG(FATAL) << "computing framework must be in [mpi, ps]. current framework: " << framework;
    return nullptr;
  }
}

} // namespace mit
