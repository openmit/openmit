#ifndef OPENMIT_FRAMEWORK_ADMM_MPI_SERVER_H_
#define OPENMIT_FRAMEWORK_ADMM_MPI_SERVER_H_

#include <memory>

#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/parameter/admm_param.h"
#include "openmit/tools/dstruct/sarray.h"

namespace mit {
/*!
 * \brief mpi model node logic for distributed machine learning
 */
class MPIServer {
  public:
    /*! \brief constructor */
    MPIServer(const mit::KWArgs & kwargs, const size_t max_dim);
    /*! \brief destructor */
    ~MPIServer();
    /*! \brief initialize model node */
    void Init(const mit::KWArgs & kwargs, const size_t max_dim);

    /*! \brief update global model */
    void Update();

    /*! \brief get global model */
    inline mit_float * Data() { 
      return theta_.data(); 
    }

    /*! \brief global model size */
    inline size_t Size() const { 
      return theta_.size(); 
    }

  private:
    /*! \brief compute global model */
    void AdmmGlobal();

  private:
    /*! \brief admm parameter */
    mit::AdmmParam admm_param_;
    /*! \brief global model */
    mit::SArray<mit_float> theta_;
}; // class MPIServer

} // namespace mit
#endif // OPENMIT_FRAMEWORK_ADMM_MPI_SERVER_H_
