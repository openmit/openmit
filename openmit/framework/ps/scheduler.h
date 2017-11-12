/*!
 *  Copyright (c) 2017 by Contributors
 *  \file scheduler.h
 *  \brief scheduler logic for parameter server
 *  \author ZhouYong
 */
#ifndef OPENMIT_FRAMEWORK_PS_SCHEDULER_H_
#define OPENMIT_FRAMEWORK_PS_SCHEDULER_H_

#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "ps/ps.h"
#include "openmit/common/arg.h"
#include "openmit/framework/ps/signal.h"
#include "openmit/tools/dstruct/dstring.h"

namespace mit {
/*!
 * \brief scheduler logic for distributed machine learning compute framework 
 */
class Scheduler {
  public:
    /*! \brief constructor */
    Scheduler(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    ~Scheduler();
    
    /*! \brief initialize scheduler */
    void Init(const mit::KWArgs & kwargs);

    /*! \brief run main logic */
    void Run();

    /*! \brief request processing logic */
    void Handle(const ps::SimpleData & recved, ps::SimpleApp * app);

  private:
    /*! \brief update metric stats info */
    void UpdateMetric(const ps::SimpleData & recved);

    /*! \brief app complete exit condition */
    void ExitCondition();

  private:
    /*! \brief ps simple app */
    std::shared_ptr<ps::SimpleApp> scheduler_;
    /*! \brief mutex */
    std::mutex mutex_;
    /*! \brief control task exit condition */
    std::condition_variable cond_;
    /*! \brief task exit tag */
    bool exit_ = false;
    /*! 
     * \brief metric info 
     *        stats worker complete numbers each epoch
     *        <datatype_metrictype, <epoch, completed_number>>
     *        "train-auc", <1, 2>
     *        "valid-logloss", <2, 2>
     */
    std::unordered_map<std::string, 
      std::unordered_map<int, int> > epoch_metric_number_;
    std::unordered_map<std::string, 
      std::unordered_map<int, float> > metric_sum_;
    /*! \brief worker complete number */
    int complete_worker_number_ = 0;
    /*! \brief server complete number */
    int complete_server_number_ = 0;
	
    DISALLOW_COPY_AND_ASSIGN(Scheduler);

}; // class Scheduler
} // namespace mit
#endif // OPENMIT_FRAMEWORK_PS_SCHEDULER_H_
