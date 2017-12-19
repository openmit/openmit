/*!
 *  Copyright (c) 2016 by Contributors
 *  \file mpi_admm.h
 *  \brief alternating-direction-multipler-method algorithm framework
 *        for distributed computing
 *  \author ZhouYong
 */
#ifndef OPENMIT_FRAMEWORK_MPI_MPI_ADMM_H_
#define OPENMIT_FRAMEWORK_MPI_MPI_ADMM_H_

#include "dmlc/io.h"
#include "openmit/common/parameter.h"
#include "openmit/framework/mpi/server.h"
#include "openmit/framework/mpi/worker.h"
#include "openmit/learner.h"
#include "openmit/tools/dstruct/sarray.h"

namespace mit {
/*!
 * \brief admm algorithm framework
 */
class MPIAdmm : public MILearner {
  public:
    /*! \brief constructor */
    MPIAdmm(const mit::KWArgs& kwargs);
    /*! \brief destructor */
    virtual ~MPIAdmm() {}
    /*! \brief get mpi admm object */
    static MPIAdmm * Get(const mit::KWArgs & kwargs) {
      return new MPIAdmm(kwargs);
    }
    /*! \brief running */
    void Run() override;

  private:
    /*! training phase by admm algorithm framework */
    void RunTrain();
    /*! predicting phase. offline predict scoring */
    void RunPredict();

  private:
    /*! \brief load model */
    void LoadModel(dmlc::Stream * fi);
    /*! \brief save predict result. [label, predict] */
    void SavePredict(dmlc::Stream * fo,
                     std::vector<std::tuple<float, float> > & preds);

  private:
    /*! \brief admm related parameter */
    mit::AdmmParam admm_param_;
    /*! \brief task related (from client) parameter */
    mit::CliParam cli_param_;
    /*! \brief global model used allreduce process. for lr */
    mit::SArray<mit_float> theta_;
    /*! \brief mpi worker */
    std::shared_ptr<mit::MPIWorker> mpi_worker_;
    /*! \brief mpi server */
    std::shared_ptr<mit::MPIServer> mpi_server_;
}; // class MPIAdmm

} // namespace mit
#endif // OPENMIT_FRAMEWORK_MPI_MPI_ADMM_H_
