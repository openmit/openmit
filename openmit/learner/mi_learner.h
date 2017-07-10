/*!
 *  Copyright (c) 2016 by Contributors
 *  \file mi_learner.h
 *  \brief Machine Intelligence Toolkits Learner
 *  \author ZhouYong
 */
#ifndef OPENMIT_LEARNER_MI_LEARNER_H_
#define OPENMIT_LEARNER_MI_LEARNER_H_

#include <string>
#include "dmlc/parameter.h"
#include "openmit/common/arg.h"

namespace mit {
/*! machine intelligence learner related parameter */
class MILearnerParam : public dmlc::Parameter<MILearnerParam> {
  public:
    /*! \brief learner tasks, 'train'/'predict'/'metric' */
    std::string task;
    /*! \brief machine learning framework. 'ps'/'mpi' */
    std::string framework;

    /*! \brief declare parameters */
    DMLC_DECLARE_PARAMETER(MILearnerParam) {
      DMLC_DECLARE_FIELD(task).set_default("train");
      DMLC_DECLARE_FIELD(framework).set_default("ps");
    }
}; // class MILearnerParam

/*!
 * \brief machine intelligence learner template for
 *        distributed machine learning tasks
 */
class MILearner {
  public:
    /*! create learner by conf args info */
    static MILearner * Create(const mit::KWArgs & kwargs);
    /*! constructor */
    MILearner() {}
    /*! destructor */
    virtual ~MILearner() {}

    /*! running */
    virtual void Run() = 0;

    inline std::string GetFrameWork() const {
      return miparam_.framework;
    }
  protected:
    /*! \brief learner parameter */
    MILearnerParam miparam_;
}; // class MITLearner
} // namespace mit
#endif // OPENMIT_LEARNER_MI_LEARNER_H_
