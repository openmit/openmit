#include "openmit/framework/admm/mpi_server.h"

namespace mit {

MPIServer::MPIServer(const mit::KWArgs & kwargs, const size_t max_dim) {
  Init(kwargs, max_dim);
}

MPIServer::~MPIServer() {
  // TODO
}

void MPIServer::Init(const mit::KWArgs & kwargs, const size_t max_dim) {
  admm_param_.InitAllowUnknown(kwargs);
  theta_.resize(max_dim, 0.0f);
  // TODO
}


void MPIServer::Run(mit_float * local,
                       mit_float * dual,
                       const size_t max_dim) {
  MiddleAggr(local, dual, max_dim);
  AdmmGlobal();
}

// TODO optimized to vector computation
void MPIServer::MiddleAggr(mit_float * local, 
                           mit_float * dual,
                           const size_t max_dim) {
  for (auto j = 0u; j < max_dim; ++j) {
    theta_[j] = *(dual + j) + admm_param_.rho * *(local + j);
  }
}

void MPIServer::ThetaUpdate() {
  auto scale = rabit::GetWorldSize() * admm_param_.rho + 1e-10;
  auto max_dim = Size();
  for (auto j = 0u; j < max_dim; ++j) {
    if (theta_[j] > admm_param_.lambda_obj) {
      theta_[j] = (theta_[j] - admm_param_.lambda_obj) / scale;
    } else if (theta_[j] < - admm_param_.lambda_obj) {
      theta_[j] = (theta_[j] + admm_param_.lambda_obj) / scale;
    } else {
      theta_[j] = 0;
    }
  }
}

void MPIServer::AdmmGlobal() {
  // AllReduce Sum
  rabit::Allreduce<rabit::op::Sum>(Data(), Size());
  if (rabit::GetRank() == 0) {
    ThetaUpdate();
  }
  // BroadCast
  rabit::Broadcast(Data(), sizeof(mit_float) * Size(), 0);
}

void MPIServer::SaveModel(dmlc::Stream * fo) {
  dmlc::ostream os(fo);
  os << "admm save model\n";
  for (auto i = 0u; i < Size(); ++i) {
    LOG(INFO) << "   i: " << i;
    os << i << "\t" << theta_[i] << "\n";
  }
  // force flush before fo destruct
  os.set_stream(nullptr);
}

void MPIServer::DebugTheta() {
  LOG(INFO) << mit::DebugStr(theta_.data(), theta_.size());
}

} // namespace mit
