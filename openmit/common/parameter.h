/*!
 *  Copyright (c) 2016 by Contributors
 *  \file parameter.h
 *  \brief parameter related to openmit computing
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_PARAMETER_H_
#define OPENMIT_COMMON_PARAMETER_H_

#include "dmlc/parameter.h"
#include "third_party/include/liblbfgs/lbfgs.h"

namespace mit {
/*!
 * \brief parameter related to client task
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
    /*! \brief model load input path (binary) */
    std::string model_in;
    /*! \brief result out path */
    std::string out_path;
    /*! \brief data format type. such as "auto"/"libsvm"/"libfm" */
    std::string data_format;
    /*! \brief whether instance data contains intercept */
    bool is_contain_intercept;
    /*! \brief nbit used to generate new_key by shifting operation */
    size_t nbit;

    // 2. task information
    /*! \brief task type. "train"/"predict" etc. default "train" */
    std::string task_type;
    /*! \brief task objective. "regression"/"binary"/"multiclass" etc. default "binary" */
    std::string objective;
    /*! \brief computational framework. "mpi"/"ps" */
    std::string framework;
    /*! \brief server global parameter update mode */
    std::string sync_mode;
    /*! \brief model. "lr", "lasso", "fm". "ffm", "mf" */
    std::string model;
    /*! \brief optimizer. "sgd", "adag", "ftrl", "als", "lbfgs", "mcmc" */
    std::string optimizer;
    /*! \brief optimizer for fm/ffm embedding learning */
    std::string optimizer_v;
    /*! \brief loss function */
    std::string loss;
    /*! \brief metric name(s) */
    std::string metric;
    /*! \brief whether train metric (many time-consuming) */
    bool is_train_metric;
    /*! \brief number of global iteration */
    uint32_t max_epoch;
    /*! \brief number of computational unit */
    uint32_t batch_size;
    /*! \brief max feature dimension id */
    uint32_t max_key;
    /*! \brief negative instances sampleing rate. [0, 1]. */
    float nsample_rate;
    /*! \brief number of threads */
    uint32_t num_thread;

    // 3. job control 
    /*! \brief transaction level. default 1 */
    size_t trans_level;
    /*! \brief is progress. default true */
    bool is_progress;
    /*! \brief job progress. batch interval numbers. default 10 */
    size_t job_progress;
    /*! \brief whether debug */
    bool debug;
    // tmp
    size_t save_peroid;

    // declare parameters
    DMLC_DECLARE_PARAMETER(CliParam) {
      DMLC_DECLARE_FIELD(train_path).set_default("");
      DMLC_DECLARE_FIELD(valid_path).set_default("");
      DMLC_DECLARE_FIELD(test_path).set_default("");
      DMLC_DECLARE_FIELD(model_in).set_default("");
      DMLC_DECLARE_FIELD(out_path).set_default("");
      DMLC_DECLARE_FIELD(data_format).set_default("libsvm")
        .describe("data format. it supports 'auto'/'libsvm'/'libfm'.");
      DMLC_DECLARE_FIELD(is_contain_intercept).set_default(false)
        .describe("whether instance data contains intercepts. such as '0:1'");
      DMLC_DECLARE_FIELD(nbit).set_default(4)
        .describe("number of bit used to key & field shifting op.");
    
      DMLC_DECLARE_FIELD(task_type).set_default("train");
      DMLC_DECLARE_FIELD(objective).set_default("objective");
      DMLC_DECLARE_FIELD(framework).set_default("ps");
      DMLC_DECLARE_FIELD(sync_mode).set_default("asp");
      DMLC_DECLARE_FIELD(model).set_default("lr");
      DMLC_DECLARE_FIELD(optimizer).set_default("ftrl");
      DMLC_DECLARE_FIELD(optimizer_v).set_default("");
      DMLC_DECLARE_FIELD(loss).set_default("logit");
      DMLC_DECLARE_FIELD(metric).set_default("auc,logloss");
      DMLC_DECLARE_FIELD(is_train_metric).set_default(true);
    
      DMLC_DECLARE_FIELD(max_epoch).set_default(2);
      DMLC_DECLARE_FIELD(batch_size).set_default(100);
      DMLC_DECLARE_FIELD(max_key).set_default(0);
      DMLC_DECLARE_FIELD(nsample_rate).set_default(0.0);
      DMLC_DECLARE_FIELD(num_thread).set_default(4);
      
      DMLC_DECLARE_FIELD(trans_level).set_default(1);
      DMLC_DECLARE_FIELD(is_progress).set_default(true);
      DMLC_DECLARE_FIELD(job_progress).set_default(10);
      DMLC_DECLARE_FIELD(debug).set_default(false);
      DMLC_DECLARE_FIELD(save_peroid).set_default(0);
    }
}; // class CliParam

/*! 
 * \brief parameter related to model
 */
struct ModelParam : public dmlc::Parameter<ModelParam> {
  /*! \brief model name */
  std::string model;
  /*! \brief max feature dimension. it is suitable for mpi/local model */
  uint32_t dim;
  /*! \brief latent vector length for fm/ffm */
  uint32_t embedding_size;
  /*! \brief field combine set */
  std::string field_combine_set;
  /*! \brief field combine pair */
  std::string field_combine_pair;
  /*! \brief nbit, merge(feature, key) shifting */
  uint32_t nbit;
  /*! \brief model initialize random method name */
  std::string random_name;
  /*! \brief random mean. applied to normal distribution */
  float random_mean;
  /*! \brief variance, it apply to normal distribution */
  float random_variance;
  /*! \brief random threshold */
  float random_threshold;

  // declare parameters 
  DMLC_DECLARE_PARAMETER(ModelParam) {
    DMLC_DECLARE_FIELD(model).set_default("lr");
    DMLC_DECLARE_FIELD(dim).set_default(1e8);
    DMLC_DECLARE_FIELD(embedding_size).set_default(4);
    DMLC_DECLARE_FIELD(field_combine_set).set_default("");
    DMLC_DECLARE_FIELD(field_combine_pair).set_default("");
    DMLC_DECLARE_FIELD(nbit).set_default(5);
    DMLC_DECLARE_FIELD(random_name).set_default("normal");
    DMLC_DECLARE_FIELD(random_mean).set_default(0.0);
    DMLC_DECLARE_FIELD(random_variance).set_default(0.0001);
    DMLC_DECLARE_FIELD(random_threshold).set_default(0.0001);
  }
}; // struct ModelParam

/*! 
 * \brief parameter related to optimizer
 */
struct OptimizerParam : public dmlc::Parameter<OptimizerParam> {
  /*! \brief optimizer name */
  std::string optimizer;
  /*! \brief learning rate */
  float lr;
  /*! \brief l1 regularation coefficient */
  float l1;
  /*! \brief l2 regularation coefficient */
  float l2;
  /*! \brief alpha for ftrl */
  float alpha;
  /*! \brief beta for ftrl */
  float beta;
  /*! \brief beta1 for adam/ftml */
  float beta1;
  /*! \brief beta2 for adam/ftml */
  float beta2;
  /*! \brief gamma decay factor for adadelta */
  float gamma;
  /*! \brief epsilon avoid denominator equals to 0 */
  float epsilon;
  /*! \brief number of corrections for lbfgs*/
  uint32_t m;
  /*! \brief maximum number of iterations for lbfgs*/
  uint32_t max_iterations;
  /*! \brief line search algorithm for lbfgs*/
  uint32_t linesearch;
  /*! \brief maximum number of trials for line search for lbfgs*/
  uint32_t max_linesearch;
  

  /*! \brief declare field */
  DMLC_DECLARE_PARAMETER(OptimizerParam) {
    DMLC_DECLARE_FIELD(optimizer).set_default("sgd");
    DMLC_DECLARE_FIELD(lr).set_default(0.001)
      .describe("learning rate. it be suitable for the optimizer that needs to initialize learning rate");
    DMLC_DECLARE_FIELD(l1).set_default(0.01);
    DMLC_DECLARE_FIELD(l2).set_default(0.01);
    DMLC_DECLARE_FIELD(alpha).set_default(0.001);
    DMLC_DECLARE_FIELD(beta).set_default(0.01);
    DMLC_DECLARE_FIELD(beta1).set_default(0.6);
    DMLC_DECLARE_FIELD(beta2).set_default(0.99);
    DMLC_DECLARE_FIELD(gamma).set_default(0.99);
    DMLC_DECLARE_FIELD(epsilon).set_default(1e-8);
    DMLC_DECLARE_FIELD(m).set_default(6);
    DMLC_DECLARE_FIELD(max_iterations).set_default(0);
    DMLC_DECLARE_FIELD(linesearch).set_default(LBFGS_LINESEARCH_DEFAULT);
    DMLC_DECLARE_FIELD(max_linesearch).set_default(20);

  } 
}; // struct OptimizerParam

/*!
 * \brief parameter related to admm distributed (decentralized) algorithm framework
 */
struct AdmmParam : public dmlc::Parameter<AdmmParam> {
  /*! \brief lambda objective function l1-norm parameter */
  float lambda_obj;
  /*! \brief rho argument lagrangians factor */
  float rho;
  /*! \brief max_dim max size of global model params */
  uint64_t max_dim;
  /*! declare parameter field */
  DMLC_DECLARE_PARAMETER(AdmmParam) {
    DMLC_DECLARE_FIELD(lambda_obj).set_default(0.05);
    DMLC_DECLARE_FIELD(rho).set_default(1)
      .describe("rho argument lagrangians factor that used to step size when dual vars update");
    DMLC_DECLARE_FIELD(max_dim).set_default(1e8);
  }
};  // class AdmmParam 

//////////// define and register parameter ////////////
// register cli parameter
inline DMLC_REGISTER_PARAMETER(CliParam);
// register model parameter 
inline DMLC_REGISTER_PARAMETER(ModelParam);
// register optimizer parameter 
inline DMLC_REGISTER_PARAMETER(OptimizerParam);
// register admm parameter
inline DMLC_REGISTER_PARAMETER(AdmmParam);

} // namespace mit
#endif // OPENMIT_COMMON_PARAMETER_H_ 
