#ifndef OPENMIT_FRAMEWORK_ADMM_MPI_SERVER_H_
#define OPENMIT_FRAMEWORK_ADMM_MPI_SERVER_H_

#include "framework/system/server.h"

namespace mit {
/*!
 * \brief mpi model node logic for distributed machine learning
 */
class MPIServer : public Server {
  public:
    void Run() override;
  private:
    // TODO

}; // class MPIServer

}

#endif // OPENMIT_FRAMEWORK_ADMM_MPI_SERVER_H_
