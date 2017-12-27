/*!
 *  Copyright (c) 2017 by Contributors
 *  \file worker.h
 *  \brief parameter server framework date node (worker) logic
 *  \author ZhouYong
 */
#ifndef OPENMIT_FRAMEWORK_PS_WORKER_H_
#define OPENMIT_FRAMEWORK_PS_WORKER_H_

#include <memory>
#include "dmlc/logging.h"
#include "dmlc/parameter.h"
#include "ps/ps.h"
#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/data.h"
#include "openmit/executor/trainer.h"
#include "openmit/executor/predictor.h"
#include "openmit/framework/ps/signal.h"
#include "openmit/metric/metric.h"
#include "openmit/tools/profiler/timer_stats.h"

namespace mit {
/*!
 * \brief worker logic for distributed machine learning compute framework 
 */
class Worker {
  public:
    /*! \brief constructor */
    Worker(const mit::KWArgs & kwargs);
    
    /*! \brief initialize worker */
    void Init(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    ~Worker();
    
    /*! \brief worker processing logic */
    void Run();

  private:
    /*! \brief training based ps */
    void RunTrain();
    
    /*! \brief predict based ps */
    void RunPredict();
    
    /*! \brief train model based on mini-batch data */
    void MiniBatch(const dmlc::RowBlock<mit_uint>& batch, 
                   std::vector<float>& batch_metric);
    
    /*! \brief key set */
    void KeySet(const dmlc::RowBlock<mit_uint>& batch, 
                std::unordered_set<mit_uint>& fset, 
                std::unordered_map<mit_uint, int>& fkv, 
                bool extra);   

  private:
    /*! \brief metric method */
    std::string Metric(mit::DMatrix * data);
    
    void MetricBatch(const dmlc::RowBlock<mit_uint> & batch, 
                     std::vector<float>& metrics_value);

    std::string MetricMsg(std::vector<float>& metrics);

  private:
    /*! \brief kv worker */
    ps::KVWorker<float> * kv_worker_;
    /*! \brief trainer */
    std::shared_ptr<mit::Trainer> trainer_;
    /*! \brief predictor */
    std::shared_ptr<mit::Predictor> predictor_;

  private:
    /*! \brief client parameter */
    mit::CliParam cli_param_;
    /*! \brief train data set */
    std::shared_ptr<mit::DMatrix> train_;
    /*! \brief train data feature set used to evaluation phase*/
    std::vector<ps::Key> train_fset_;
    /*! \brief validation data set */
    std::shared_ptr<mit::DMatrix> valid_;
    /*! \brief validation data set */
    std::shared_ptr<mit::DMatrix> test_;
    /*! \brief timer stats */
    mit::STATS stats;
}; // class Worker
} // namespace mit

#endif // OPENMIT_FRAMEWORK_PS_WORKER_H_
