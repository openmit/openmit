/*!
 *  Copyright (c) 2017 by Contributors
 *  \file ps.h
 *  \brief parameter server distributed computation framework
 *  \author ZhouYong
 */
#ifndef OPENMIT_FRAMEWORK_PS_H_
#define OPENMIT_FRAMEWORK_PS_H_

#include "openmit/framework/scheduler.h"
#include "openmit/framework/server.h"
#include "openmit/framework/worker.h"

namespace mit {
/*!
 * \brief parameter server framework for
 *        distributed machine learning tasks.
 */
class PS {
  public:
    /*! \brief constructor */
    PS(const mit::KWArgs & kwargs);
    /*! \brief destructor */
    ~PS() {}
    /*! \brief running */
    void Run();
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
#endif // OPENMIT_FRAMEWORK_PS_H_
