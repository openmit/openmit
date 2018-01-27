/*!
 *  Copyright (c) 2016 by Contributors
 *  \file learner.h
 *  \brief machine intelligence learner
 *  \author ZhouYong
 */

#ifndef OPENMIT_LEARNER_H_
#define OPENMIT_LEARNER_H_ 

#include <string>
#include "openmit/common/arg.h"
#include "openmit/framework/ps.h"

namespace mit {
/*!
 * \brief machine intelligence learner for (distributed) machine learning tasks
 */
class MILearner {
public:
  /*! brief constructor */
  MILearner(const mit::KWArgs& kwargs);
  /*! brief destructor */
  ~MILearner();
  /*! brief running machine intelligence task */
  void Run();

private:
  mit::PS* ps_;
}; // class MILearner 

MILearner::MILearner(const mit::KWArgs& kwargs) {
  ps_ = new mit::PS(kwargs);
}

MILearner::~MILearner() {
  if (ps_) {
    delete ps_; ps_ = nullptr;
  }
}

void MILearner::Run() {
  ps_->Run();
}

} // namespace mit
#endif // OPENMIT_LEARNER_H_ 
