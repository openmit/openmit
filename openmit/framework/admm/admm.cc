#include "dmlc/logging.h"
#include "dmlc/timer.h"
#include "rabit/rabit.h"
#include "openmit/framework/admm/admm.h"

namespace mit {

DMLC_REGISTER_PARAMETER(AdmmParam);

Admm::Admm(const mit::KWArgs & kwargs) {
  this->miparam_.InitAllowUnknown(kwargs);
  param_.InitAllowUnknown(kwargs);
  LOG(INFO) << "Admm param_.rho: " << param_.rho;
}

void Admm::Run() {
  rabit::Init(0, new char*[1]);
  double startTime = dmlc::GetTime();
  if (this->miparam_.task == "train") {
    RunTrain();
  } else if (this->miparam_.task == "predict") {
    RunPredict();
  } else {
    LOG(ERROR) 
      << this->miparam_.task 
      << " is not in [train, predict].";
  }
  double endTime = dmlc::GetTime();
  rabit::TrackerPrintf("@node[%d] [OpenMIT-ADMM] \
      The total time of the task %s is %g min \n", 
      rabit::GetRank(), 
      this->miparam_.task.c_str(), 
      (endTime-startTime)/60);

  rabit::Finalize();
}

void Admm::RunTrain() {
  LOG(INFO) << "Admm::RunTrain() ...";
}

void Admm::RunPredict() {
  LOG(INFO) << "Admm::RunPredict() ...";
}

} // namespace mit
