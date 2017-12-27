#include <stdlib.h>
#include <unistd.h>
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
  dmlc::InitLogging("openmit-ps\0");

  LOG(INFO) << "ps task begin. ";
  ps::Start();
  std::string nodeinfo = ps::IsServer() ? "@server[" : (ps::IsWorker() ? "@worker[" : "@scheduler[");
  nodeinfo += std::to_string(ps::MyRank()) + "] ";

  LaunchScheduler();
  LaunchServer();
  LaunchWorker();

  nodeinfo += "task has completed.";
  LOG(INFO) << nodeinfo << " start callback finalize op.";
  ps::Finalize(false);
  LOG(INFO) << nodeinfo << " finalize successfully!!!";
}

void PS::LaunchScheduler() {
  std::string nodeinfo = ps::IsServer() ? "@server " : (ps::IsWorker() ? "@worker " : "@scheduler ");
  if (!ps::IsScheduler()) return;
  LOG(INFO) << nodeinfo << "launch scheduler begin.";
  auto scheduler = new mit::Scheduler(kwargs_);
  ps::RegisterExitCallback([scheduler]() {
    std::string nodeinfo = ps::IsServer() ? "@server " : (ps::IsWorker() ? "@worker " : "@scheduler ");
    LOG(INFO) << nodeinfo << "delete scheduler begin.";
    delete scheduler; 
    LOG(INFO) << nodeinfo << "delete scheduler done";
  });
  scheduler->Run();
}

void PS::LaunchServer() {
  if (!ps::IsServer()) return;
  std::string nodeinfo = ps::IsServer() ? "@server[" : (ps::IsWorker() ? "@worker[" : "@scheduler[");
  LOG(INFO) << nodeinfo << "launch server begin.";
  auto server = new mit::Server(kwargs_);
  ps::RegisterExitCallback([server]() { 
    std::string nodeinfo = ps::IsServer() ? "@server " : (ps::IsWorker() ? "@worker " : "@scheduler ");
    LOG(INFO) << nodeinfo << "delete server begin.";
    delete server; 
    LOG(INFO) << nodeinfo << "delete server done.";
  });
  server->Run();
}

void PS::LaunchWorker() {
  std::string nodeinfo = ps::IsServer() ? "@server[" : (ps::IsWorker() ? "@worker[" : "@scheduler[");
  if (!ps::IsWorker()) return;
  LOG(INFO) << nodeinfo << "launch worker begin.";
  auto worker = new mit::Worker(kwargs_);
  ps::RegisterExitCallback([worker]() {
    std::string nodeinfo = ps::IsServer() ? "@server " : (ps::IsWorker() ? "@worker " : "@scheduler ");
    LOG(INFO) << nodeinfo << "delete worker begin.";
    delete worker;
    LOG(INFO) << nodeinfo << "delete worker done.";
  });
  worker->Run();
}

} // namespace mit
