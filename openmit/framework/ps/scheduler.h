/*!
 *  Copyright (c) 2017 by Contributors
 *  \file scheduler.h
 *  \brief scheduler logic for parameter server
 *  \author ZhouYong
 */
#ifndef OPENMIT_FRAMEWORK_PS_SCHEDULER_H_
#define OPENMIT_FRAMEWORK_PS_SCHEDULER_H_

#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <condition_variable>

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


    void Run();

    /*! \brief scheduler processing logic */
    void Handle(const ps::SimpleData & recved, ps::SimpleApp * app);

  private:
    void UpdateMetric(const ps::SimpleData & recved);
    
    void MetricInfo(const std::string & data_type, 
                               const std::string & metric_type,
                               int epoch, 
                               float metric_value);
    void ExitCondition();

  private:
    /*! \brief ps simple app */
    std::shared_ptr<ps::SimpleApp> scheduler_;

    std::mutex mutex_;

    std::condition_variable cond_;

    bool exit_ = false;
    
    /*! 
     * \brief metric info 
     *        stats worker complete numbers each epoch
     *        <type, <epoch, completed_number>>
     *        metric type: "logloss"/"auc"
     */
    std::unordered_map<std::string, 
      std::unordered_map<int, int> > epoch_metric_number_train_;
    std::unordered_map<std::string, 
      std::unordered_map<int, float> > metric_sum_train_;
    std::unordered_map<std::string, 
      std::unordered_map<int, int> > epoch_metric_number_eval_;
    std::unordered_map<std::string, 
      std::unordered_map<int, float> > metric_sum_eval_;
    /*! \brief worker complete number */
    int complete_worker_number_ = 0;
    /*! \brief server complete number */
    int complete_server_number_ = 0;
	
    DISALLOW_COPY_AND_ASSIGN(Scheduler);

}; // class Scheduler
} // namespace mit

#endif // OPENMIT_FRAMEWORK_PS_SCHEDULER_H_
