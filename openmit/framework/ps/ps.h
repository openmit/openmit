/*!
 *  Copyright (c) 2017 by Contributors
 *  \file parameter_server.h
 *  \brief parameter server distributed computation framework
 *  \author ZhouYong
 */
#ifndef OPENMIT_FRAMEWORK_PS_PS_H_
#define OPENMIT_FRAMEWORK_PS_PS_H_

#include "openmit/learner.h"
#include "openmit/framework/ps/scheduler.h"
#include "openmit/framework/ps/server.h"
#include "openmit/framework/ps/worker.h"

namespace mit {
/*!
 * \brief parameter server framework for
 *        distributed machine learning tasks.
 */
class PS : public MILearner {
  public:
    /*! \brief constructor */
    PS(const mit::KWArgs & kwargs);
    /*! \brief destructor */
    virtual ~PS() {}
    /*! \brief get parameter server object */
    static PS * Get(const mit::KWArgs & kwargs) {
      return new PS(kwargs);
    }
    /*! \brief running */
    void Run() override;
  private:
    /*! \brief scheduler launcher */
    void LaunchScheduler();
    /*! \brief server launcher */
    void LaunchServer();
    /*! \brief worker launcher */
    void LaunchWorker();
  private:
    /*! \brief kwargs */
    mit::KWArgs kwargs_;
    /*! \brief client parameter */
    CliParam cli_param_;
}; // class PS

} // namespace mit
#endif // OPENMIT_FRAMEWORK_PS_PS_H_
