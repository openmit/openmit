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
#include "openmit/common/data/data.h"
#include "openmit/engine/trainer.h"
#include "openmit/engine/predictor.h"
#include "openmit/metric/metric.h"
#include "openmit/framework/ps/signal.h"

namespace mit {
/*!
 * \brief worker related parameter
 */
class WorkerParam : public dmlc::Parameter<WorkerParam> {
  public:
    /*! \brief task type. "train"/"predict" etc. default: "train" */
    std::string task_type;
    /*! \brief data_format */
    std::string data_format;
    /*! \brief train data path */
    std::string train_path;
    /*! \brief evaluation data path */
    std::string valid_path;
    /*! \brief test data path */
    std::string test_path;
    /*! \breif predict output path */
    std::string predict_out;
    /*! \brief size of batch data */
    int batch_size;
    /*! \brief max number of iteration */
    uint32_t max_epoch;
    /*! \brief whether shuffle data before training */
    bool is_shuffle;
    /*! \brief metric method, 'logloss', 'auc', ... */
    std::string metric;
    /*! \brief peroid saved model */
    int save_peroid;
    /*! \brief embedding_size */
    size_t embedding_size;
    /*! \brief field number for ffm */
    size_t field_num;

    /*! \brief declare parameters */
    DMLC_DECLARE_PARAMETER(WorkerParam) {
      DMLC_DECLARE_FIELD(task_type).set_default("train");
      DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
      DMLC_DECLARE_FIELD(train_path).set_default("");
      DMLC_DECLARE_FIELD(valid_path).set_default("");
      DMLC_DECLARE_FIELD(test_path).set_default("");
      DMLC_DECLARE_FIELD(predict_out).set_default("");
      DMLC_DECLARE_FIELD(batch_size).set_default(100);
      DMLC_DECLARE_FIELD(max_epoch).set_default(10);
      DMLC_DECLARE_FIELD(is_shuffle).set_default(false);
      DMLC_DECLARE_FIELD(metric).set_default("auc");
      DMLC_DECLARE_FIELD(save_peroid).set_default(0);
      DMLC_DECLARE_FIELD(embedding_size).set_default(4);
      DMLC_DECLARE_FIELD(field_num).set_default(10);
    }
}; // class WorkerParam

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

    /*! model effect evaluation */
    float Metric(mit::DMatrix * data, std::vector<ps::Key> & feat_set);

  private:
    /*! \brief training based ps */
    void RunTrain();
    /*! \brief predict based ps */
    void RunPredict();
    
  private:
    /*! \brief initialize feature set if size of feature < 1e6 */
    void InitFSet(mit::DMatrix * data, std::vector<ps::Key> * feat_set);
    /*! \brief train model based on mini-batch data */
    void MiniBatch(const dmlc::RowBlock<mit_uint> & batch);

  private:
    /*! \brief kv worker */
    ps::KVWorker<float> * kv_worker_;
    /*! \brief trainer */
    std::shared_ptr<mit::Trainer> trainer_;
    /*! \brief predictor */
    std::shared_ptr<mit::Predictor> predictor_;

  private:
    /*! \brief server parameter info */
    mit::WorkerParam param_;
    /*! \brief train data set */
    std::shared_ptr<mit::DMatrix> train_set_;
    /*! \brief train data feature set used to evaluation phase*/
    std::vector<ps::Key> train_fset_;
    /*! \brief validation data set */
    std::shared_ptr<mit::DMatrix> valid_set_;
    /*! \brief validation data feature set */
    std::vector<ps::Key> valid_fset_;  
    /*! \brief validation data set */
    std::shared_ptr<mit::DMatrix> test_set_;
}; // class Worker
} // namespace mit

#endif // OPENMIT_FRAMEWORK_PS_WORKER_H_
