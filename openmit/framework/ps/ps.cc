#include <stdlib.h>
#include "openmit/framework/ps/ps.h"
#include "ps/ps.h"

namespace mit {

PS::PS(const mit::KWArgs & kwargs) {
  kwargs_ = kwargs;
  cli_param_.InitAllowUnknown(kwargs);
  // feature max feature dimension
  uint64_t max_key = cli_param_.max_key > 0 
    ? cli_param_.max_key : std::numeric_limits<uint64_t>::max();
  // register env variable for ps-lite
  setenv("DMLC_MAX_DIMENSION", std::to_string(max_key).c_str(), 1);
  LOG(INFO) << "cli_param_.max_key: " << getenv("DMLC_MAX_DIMENSION");
}

void PS::Run() {
  LOG(INFO) << "ps task begin.";
  ps::Start();
  
  LaunchScheduler();
  LaunchServer();
  LaunchWorker();

  std::string exitinfo = (ps::IsServer() ? 
    "@server[" : (ps::IsWorker() ? "@worker[" : "@scheduler[")) 
    + std::to_string(ps::MyRank()) + "] task has completed.";
  LOG(INFO) << exitinfo << " start callback finalize op.";
  ps::Finalize(false);
  LOG(INFO) << exitinfo << " finalize successfully!!!";
}

void PS::LaunchScheduler() {
  if (!ps::IsScheduler()) return;
  LOG(INFO) << "launch scheduler begin.";
  auto scheduler = new mit::Scheduler(kwargs_);
  ps::RegisterExitCallback([scheduler]() { 
    delete scheduler; 
    LOG(INFO) << "delete scheduler done";
  });
  scheduler->Run();
}

void PS::LaunchServer() {
  if (!ps::IsServer()) return;
  LOG(INFO) << "launch server begin.";
  auto server = new mit::Server(kwargs_);
  ps::RegisterExitCallback([server]() { 
    delete server; 
    LOG(INFO) << "delete server done.";
  });
  server->Run();
}

void PS::LaunchWorker() {
  if (!ps::IsWorker()) return;
  LOG(INFO) << "launch worker begin.";
  auto worker = new mit::Worker(kwargs_);
  ps::RegisterExitCallback([worker]() {
    delete worker;
    LOG(INFO) << "delete worker done.";
  });
  worker->Run();
}

} // namespace mit
