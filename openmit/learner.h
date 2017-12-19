/*!
 *  Copyright (c) 2016 by Contributors
 *  \file data.h
 *  \brief data instance set structure
 *  \author ZhouYong
 */

#ifndef OPENMIT_LEARNER_H_
#define OPENMIT_LEARNER_H_ 

#include <string>
#include "openmit/common/arg.h"

namespace mit {
/*!
 * \brief machine intelligence learner for (distributed) machine learning tasks
 */
class MILearner {
public:
  /*! brief create learner by conf args info */
  static MILearner* Create(const mit::KWArgs& kwargs);
  /*! brief constructor */
  MILearner() {}
  /*! brief destructor */
  virtual ~MILearner() {}
  /*! brief running machine intelligence task */
  virtual void Run() = 0;
}; // class MILearner 

} // namespace mit
#endif // OPENMIT_LEARNER_H_ 
