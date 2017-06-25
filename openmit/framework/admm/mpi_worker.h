#ifndef OPENMIT_FRAMEWORK_ADMM_MPI_WORKER_H_
#define OPENMIT_FRAMEWORK_ADMM_MPI_WORKER_H_

#include <memory>

#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/data/data.h"
#include "openmit/common/parameter/parameter.h"
#include "openmit/tools/dstruct/sarray.h"

namespace mit {
/*!
 * \brief mpi data node logic for distributed machine learning
 */
class MPIWorker {
  public:
    /*! \brief */
    MPIWorker() {}
    /*! \brief initialize data node */
    void Init(const mit::KWArgs & kwargs);
    void Run();
  private:
    mit::CliParam param_;
    /*! \brief local model parameter. for lr */
    mit::SArray<mit_float> weight_;
    /*! \brief dual variables parameter */
    mit::SArray<mit_float> dual_;
    /*! \brief task related data set */
    std::shared_ptr<mit::DMatrix> train_set_;
    std::shared_ptr<mit::DMatrix> valid_set_;
    std::shared_ptr<mit::DMatrix> test_set_;
}; // class MPIWorker
} // namespace mit

#endif // OPENMIT_FRAMEWORK_ADMM_MPI_WORKER_H_
