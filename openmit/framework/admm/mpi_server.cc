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


void MPIServer::Update() {
  // TODO
  AdmmGlobal();
}

void MPIServer::AdmmGlobal() {
  // TODO
}


} // namespace mit
