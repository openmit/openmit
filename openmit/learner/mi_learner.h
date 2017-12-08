/*!
 *  Copyright (c) 2016 by Contributors
 *  \file mi_learner.h
 *  \brief machine intelligence toolkits learner
 *  \author ZhouYong
 */
#ifndef OPENMIT_LEARNER_MI_LEARNER_H_
#define OPENMIT_LEARNER_MI_LEARNER_H_

//#include <memory>
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
    /*! brief create learner by conf args info */
    static MILearner * Create(const mit::KWArgs& kwargs);
    
    /*! brief constructor */
    MILearner() {}
    
    /*! brief destructor */
    virtual ~MILearner() {}

    /*! brief running machine intelligence task */
    virtual void Run() = 0;
}; // class MITLearner

} // namespace mit
#endif // OPENMIT_LEARNER_MI_LEARNER_H_
