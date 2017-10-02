#ifndef OPENMIT_FRAMEWORK_MPI_WORKER_H_
#define OPENMIT_FRAMEWORK_MPI_WORKER_H_

#include <memory>

#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/data/data.h"
#include "openmit/common/parameter/admm_param.h"
#include "openmit/common/parameter/cli_param.h"
#include "openmit/models/model.h"
#include "openmit/optimizer/optimizer.h"
#include "openmit/tools/dstruct/sarray.h"

namespace mit {
/*!
 * \brief mpi data node logic for distributed machine learning
 */
class MPIWorker {
  public:
    /*! \brief constructor */
    MPIWorker(const mit::KWArgs & kwargs);
    /*! \brief destructor */
    ~MPIWorker();
    /*! \brief initialize data node */
    void Init(const mit::KWArgs & kwargs);

    /*! 
     * \brief compute local model using global model and partial-data
     */
    void Run(mit_float * global, const size_t size, const size_t epoch);

    /*! 
     * \brief update dual parameter using global model. 
     *        update formula: 
     *          dual_[j] <- dual_[j] + \rho * (w_[j] - \theta[j])
     */
    void UpdateDual(mit_float * global, const size_t size);

    /*! \brief get local model */
    inline mit_float * Data() { return weight_.data(); }

    /*! \brief dual info */
    inline mit_float * Dual() { return dual_.data(); }

    /*! \brief size */
    inline size_t Size() const {
      CHECK_EQ(weight_.size(), dual_.size()) 
        << "mpi_worker weight_.size != dual_.size";
      return weight_.size();
    }
    
    /*! \brief debug mpi worker */
    void Debug();

  private:
    /*! \brief mini-batch computation */
    void MiniBatch(const dmlc::RowBlock<mit_uint> & batch);
    
    /*! \brief debug weight_ */
    void DebugWeight();
    
    /*! \brief debug dual_ */
    void DebugDual();

  private:
    /*! \brief model */
    std::shared_ptr<mit::Model> model_;
    /*! \brief optimizer */
    std::shared_ptr<mit::Opt> opt_;
    /*! \brief client parameter */
    mit::CliParam cli_param_;
    /*! \brief algorithm framework params */
    mit::AdmmParam admm_param_;
    /*! \brief local model parameter. for lr */
    mit::SArray<mit_float> weight_;
    /*! \brief dual variables parameter */
    mit::SArray<mit_float> dual_;
    /*! \brief train data  */
    std::shared_ptr<mit::DMatrix> train_;
    /*! \brief valid data  */
    std::shared_ptr<mit::DMatrix> valid_;
    /*! \brief test data for predict task */
    std::shared_ptr<mit::DMatrix> test_;

}; // class MPIWorker
} // namespace mit

#endif // OPENMIT_FRAMEWORK_MPI_WORKER_H_
