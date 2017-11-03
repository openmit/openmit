/*!
 *  Copyright 2016 by Contributors
 *  \file cli_param.h
 *  \brief client configure parameter
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_PARAMETER_CLI_PARAM_H_
#define OPENMIT_COMMON_PARAMETER_CLI_PARAM_H_

#include "dmlc/parameter.h"
#include "openmit/common/base.h"

namespace mit {
/*!
 * \brief client related parameter
 */
class CliParam : public dmlc::Parameter<CliParam> {
  public:
    // 1. path and data information
    /*! \brief train data path */
    std::string train_path;
    /*! \brief valid data path */
    std::string valid_path;
    /*! \brief test data path */
    std::string test_path;
    /*! \brief predict result output path */
    std::string predict_out;
    /*! \brief model dump output path */
    std::string model_dump;
    /*! \brief model binary output path */
    std::string model_binary;
    /*! \brief model load input path (binary) */
    std::string model_in;
    /*! \brief data format type. such as "auto" "libsvm", "libfm" */
    std::string data_format;
    /*! \brief nbit used to generate new_key by shifting operation. "libfm" */
    size_t nbit;

    // 2. task information
    /*! \brief task type. "train", "predict" etc. default "train" */
    std::string task_type;
    /*! \brief computational framework. "mpi"/"ps" */
    std::string framework;
    /*! \brief server global parameter update mode. "asp", "bsp", "ssp" */
    std::string sync_mode;
    /*! \brief model. "lr", "lasso", "fm". "ffm", "mf" */
    std::string model;
    /*! \brief optimizer. "sgd", "adag", "ftrl", "als", "lbfgs", "mcmc" */
    std::string optimizer;
    /*! \brief optimizer for fm/ffm embedding learning */
    std::string optimizer_v;
    /*! \brief loss function */
    std::string loss;
    /*! \brief metric */
    std::string metric;
    /*! \brief number of global iteration. it equals to number of global_weight updated */
    uint32_t max_epoch;
    /*! \brief number of splitted data block at each worker node. default 10 */
    uint32_t batch_size;
    /*! \brief negative instances sampleing rate. [0, 1]. */
    float nsample_rate;

    // 3. job control 
    /*! \brief transaction level. default 1 */
    size_t trans_level;
    /*! \brief is progress. default true */
    bool is_progress;
    /*! \brief job progress. batch interval numbers. default 10 */
    size_t job_progress;
    /*! \brief whether debug */
    bool debug;

    /*! \brief weight threshold value. 1e-8 */
    double w_minv;
    size_t embedding_size;


    // tmp
    size_t save_peroid;

    // declare parameters
    DMLC_DECLARE_PARAMETER(CliParam) {
      DMLC_DECLARE_FIELD(train_path).set_default("");
      DMLC_DECLARE_FIELD(valid_path).set_default("");
      DMLC_DECLARE_FIELD(test_path).set_default("");
      DMLC_DECLARE_FIELD(predict_out).set_default("");
      DMLC_DECLARE_FIELD(model_dump).set_default("");
      DMLC_DECLARE_FIELD(model_binary).set_default("");
      DMLC_DECLARE_FIELD(model_in).set_default("");
      DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
      DMLC_DECLARE_FIELD(nbit).set_default(4);

      DMLC_DECLARE_FIELD(task_type).set_default("train");
      DMLC_DECLARE_FIELD(framework).set_default("ps");
      DMLC_DECLARE_FIELD(sync_mode).set_default("asp");
      DMLC_DECLARE_FIELD(model).set_default("lr");
      DMLC_DECLARE_FIELD(optimizer).set_default("ftrl");
      DMLC_DECLARE_FIELD(optimizer_v).set_default("");
      DMLC_DECLARE_FIELD(loss).set_default("logit");
      DMLC_DECLARE_FIELD(metric).set_default("auc,logloss");
      
      DMLC_DECLARE_FIELD(max_epoch).set_default(2);
      DMLC_DECLARE_FIELD(batch_size).set_default(100);
      DMLC_DECLARE_FIELD(nsample_rate).set_default(0.0);
      
      DMLC_DECLARE_FIELD(trans_level).set_default(1);
      DMLC_DECLARE_FIELD(is_progress).set_default(true);
      DMLC_DECLARE_FIELD(job_progress).set_default(10);
      DMLC_DECLARE_FIELD(debug).set_default(false);

      DMLC_DECLARE_FIELD(w_minv).set_default(1e-8);

      DMLC_DECLARE_FIELD(save_peroid).set_default(0);
      DMLC_DECLARE_FIELD(embedding_size).set_default(4);
    }
}; // class CliParam

} // namespace mit

#endif // OPENMIT_COMMON_PARAMETER_CLI_PARAM_H_
