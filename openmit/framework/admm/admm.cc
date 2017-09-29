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
      << this->miparam_.task << " is not in [train, predict].";
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
  for (auto iter = 0u; iter < cli_param_.max_epoch; ++iter) {
    LOG(INFO) << "Admm::RunTrain iter: " << iter;
    // learning 
    mpi_worker_->Run(mpi_server_->Data(), mpi_server_->Size());
    mpi_server_->Run(mpi_worker_->Data(), mpi_worker_->Dual(), mpi_worker_->Size());
    mpi_worker_->UpdateDual(mpi_server_->Data(), mpi_server_->Size());
    
    if (cli_param_.debug) { 
      mpi_worker_->Debug(); 
      mpi_server_->DebugTheta();
    }
    // metric TODO
  }
  if (rabit::GetRank() == 0) {
    LOG(INFO) << "Admm::RunTrain SaveModel begin";
    std::unique_ptr<dmlc::Stream> fo(
      dmlc::Stream::Create(cli_param_.model_dump.c_str(), "w"));
    mpi_server_->SaveModel(fo.get());
    LOG(INFO) << "Admm::RunTrain SaveModel done ";
  }
  LOG(INFO) << "Admm::RunTrain done ";
}

void Admm::RunPredict() {
  LOG(INFO) << "Admm::RunPredict() ...";
}

} // namespace mit
