#ifndef OPENMIT_FRAMEWORK_ADMM_MPI_SERVER_H_
#define OPENMIT_FRAMEWORK_ADMM_MPI_SERVER_H_

#include <memory>

#include "rabit/rabit.h"
#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/parameter/admm_param.h"
#include "openmit/tools/dstruct/sarray.h"

namespace mit {
/*!
 * \brief mpi global model node logic for distributed machine learning
 */
class MPIServer {
  public:
    /*! \brief constructor */
    MPIServer(const mit::KWArgs & kwargs, 
              const size_t max_dim);
    
    /*! \brief destructor */
    ~MPIServer();
    
    /*! \brief initialize model node */
    void Init(const mit::KWArgs & kwargs, 
              const size_t max_dim);

    /*! \brief update global model */
    void Run(mit_float * local, 
                mit_float * dual, 
                const size_t max_dim);

    /*! \brief get global model */
    inline mit_float * Data() { 
      return theta_.data(); 
    }

    /*! \brief global model size */
    inline size_t Size() const { 
      return theta_.size(); 
    }

    /*! \brief save global model */
    void SaveModel(dmlc::Stream * fo);
    
    /*! \brief debug theta_ */
    void DebugTheta();

  private:
    /*
     * \brief global model middle value. formula:
     *        (beta_t[j] + \rho * w_t[j])
     */
    void MiddleAggr(mit_float * local, 
                    mit_float * dual,
                    const size_t max_dim);

    /* \brief global theta update */
    void ThetaUpdate();

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
