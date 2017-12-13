/*!
 *  Copyright (c) 2016 by Contributors
 *  \file mi_learner.h
 *  \brief Machine Intelligence Toolkits Learner
 *  \author ZhouYong
 */
#ifndef OPENMIT_LEARNER_MI_LEARNER_H_
#define OPENMIT_LEARNER_MI_LEARNER_H_

#include <memory>
#include <string>
#include "dmlc/parameter.h"
#include "openmit/common/arg.h"

namespace mit {
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
}; // class MITLearner
} // namespace mit
#endif // OPENMIT_LEARNER_MI_LEARNER_H_
