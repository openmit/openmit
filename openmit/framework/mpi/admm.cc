#include "dmlc/logging.h"
#include "dmlc/timer.h"
#include "rabit/rabit.h"
#include "openmit/framework/mpi/admm.h"
#include "openmit/tools/monitor/transaction.h"

namespace mit {

Admm::Admm(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

void Admm::Init(const mit::KWArgs & kwargs) {
  rabit::Init(0, new char*[1]);
  this->miparam_.InitAllowUnknown(kwargs);
  admm_param_.InitAllowUnknown(kwargs);
  cli_param_.InitAllowUnknown(kwargs);
  mpi_worker_.reset(new MPIWorker(kwargs));
  mpi_server_.reset(new MPIServer(kwargs, mpi_worker_->Size()));
}

void Admm::Run() {
  double startTime = dmlc::GetTime();
  if (this->miparam_.task_type == "train") {
    std::unique_ptr<Transaction> trans(
        new Transaction(0, "mpi", "train", true));
    RunTrain();
    Transaction::End(trans.get());
  } else if (this->miparam_.task_type == "predict") {
    std::unique_ptr<Transaction> trans(
        new Transaction(0, "mpi", "predict", true));
    RunPredict();
    Transaction::End(trans.get());
  } else {
    LOG(ERROR) 
      << this->miparam_.task_type << " not in [train, predict].";
  }
  double endTime = dmlc::GetTime();
  rabit::TrackerPrintf("@worker[%d] [OpenMIT-ADMM] \
      The total time of the task %s is %g min \n", 
      rabit::GetRank(), 
      this->miparam_.task_type.c_str(), 
      (endTime-startTime)/60);
  rabit::Finalize();
}

void Admm::RunTrain() {
  for (auto iter = 0u; iter < cli_param_.max_epoch; ++iter) {
    // learning 
    mpi_worker_->Run(mpi_server_->Data(), mpi_server_->Size(), iter + 1);
    mpi_server_->Run(mpi_worker_->Data(), mpi_worker_->Dual(), mpi_worker_->Size());
    mpi_worker_->UpdateDual(mpi_server_->Data(), mpi_server_->Size());
    
    if (cli_param_.debug) { 
      mpi_worker_->Debug(); 
      mpi_server_->DebugTheta();
    }
    // metric TODO
    if (rabit::GetRank() == rabit::GetWorldSize() - 1) {
      // TODO  metric allreduce & broadcast
      LOG(INFO) << "finished " << iter+1 << "-th epoch. metric ...";
    }
  }
  if (rabit::GetRank() == 0) {
    std::unique_ptr<dmlc::Stream> fo(
      dmlc::Stream::Create(cli_param_.model_dump.c_str(), "w"));
    mpi_server_->SaveModel(fo.get());
  }
}

void Admm::RunPredict() {
  LOG(INFO) << "Admm::RunPredict() ...";
}

} // namespace mit
