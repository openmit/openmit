#include "dmlc/logging.h"
#include "dmlc/timer.h"
#include "rabit/rabit.h"
#include "openmit/framework/admm/admm.h"

namespace mit {

Admm::Admm(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

void Admm::Init(const mit::KWArgs & kwargs) {
  this->miparam_.InitAllowUnknown(kwargs);
  admm_param_.InitAllowUnknown(kwargs);
  cli_param_.InitAllowUnknown(kwargs);
  LOG(INFO) << "Admm param_.rho: " << admm_param_.rho;
  mpi_worker_.reset(new MPIWorker(kwargs));
  mpi_server_.reset(new MPIServer(kwargs, mpi_worker_->Size()));
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
  for (auto iter = 0u; iter < cli_param_.max_epoch; ++iter) {
    // learning 
    LOG(INFO) << "Admm::RunTrain() 1 ...";
    mpi_worker_->Update(mpi_server_->Data(), mpi_server_->Size());
    LOG(INFO) << "Admm::RunTrain() 2 ...";
    mpi_server_->Update();
    LOG(INFO) << "Admm::RunTrain() 3 ...";
    mpi_worker_->UpdateDual(mpi_server_->Data(), mpi_server_->Size());
    LOG(INFO) << "Admm::RunTrain() 4 ...";

    // metric
  }

}

void Admm::RunPredict() {
  LOG(INFO) << "Admm::RunPredict() ...";
}

} // namespace mit
