#include "openmit/framework/ps/parameter_server.h"
#include "ps/ps.h"

namespace mit {

DMLC_REGISTER_PARAMETER(PSParam);

PS::PS(const mit::KWArgs & kwargs) {
  kwargs_ = kwargs;
  this->miparam_.InitAllowUnknown(kwargs);
  param_.InitAllowUnknown(kwargs);
  //LOG(INFO) << "PS::PS()~ framework: " << param_.framework;
}

void PS::Run() {
  LOG(INFO) << "PS::Run() beginning";
  ps::Start();
  LaunchScheduler();
  LaunchServer();
  LaunchWorker();
  ps::Finalize(true);
  LOG(INFO) << "PS::Run finalize!";
  // TODO
}

void PS::LaunchScheduler() {
  if (!ps::IsScheduler()) return;
  auto scheduler = new mit::Scheduler(kwargs_);
  ps::RegisterExitCallback([scheduler]() { delete scheduler; });
}

void PS::LaunchServer() {
  if (!ps::IsServer()) return;
  auto server = new mit::Server(kwargs_);
  ps::RegisterExitCallback([server]() { delete server; });
}

void PS::LaunchWorker() {
  if (!ps::IsWorker()) return;
  mit::Worker worker(kwargs_);
  worker.Run();
}

} // namespace mit
