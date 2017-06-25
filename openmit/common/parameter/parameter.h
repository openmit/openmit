#ifndef OPENMIT_COMMON_PARAMETER_PARAMETER_H_
#define OPENMIT_COMMON_PARAMETER_PARAMETER_H_

#include "dmlc/parameter.h"
#include "openmit/common/base.h"

namespace mit {
/*!
 * \brief client related parameter
 */
class CliParam : public dmlc::Parameter<CliParam> {
  public:
    /*! \brief task type. "train", "predict" etc. default "train" */
    std::string task;
    /*! \brief model. "lr", "lasso", "fm". "ffm", "mf" */
    std::string model;
    /*! \brief optimizer. "sgd", "adag", "ftrl", "als", "lbfgs", "mcmc" */
    std::string optimizer;
    /*! \brief server global parameter update mode. "async","sync","admm" (default admm) */
    std::string sync_mode;
    /*! \brief master "local", "yarn", default yarn */
    std::string master;
    /* ! \brief data format type. such as "auto" "libsvm" */
    std::string data_format;
    /*! \brief train data path */
    std::string train_path;
    /*! \brief valid data path */
    std::string valid_path;
    /*! \brief test data path */
    std::string test_path;
    /*! \brief model file out path */
    std::string model_out;
    /*! \brief number of global iteration. it equals to number of global_weight updated */
    uint32_t max_epoch;
    /*! \brief number of local param update at each global iteration */
    int passes;
    /*! \brief number of splitted data block at each worker node */
    int batch_size;
    /*! \brief feature dimension */
    uint32_t dim;
    /*! \brief number of feature field for ffm model */
    size_t field_num;
    int featgrp_nbits;
    /*! \brief latent vector length for fm/ffm */
    int k;
    /*! \brief negative instances sampleing rate. [0, 1]. */
    float nsample_rate;
    /*! \brief alpha for ftrl learning rate */
    float alpha;
    /*! \brief beta for ftrl learning rate */
    float beta;
    /*! \brief l1 regularation */
    float l1;
    /*! \brief l2 regularation */
    float l2;
    /*! \brief is debug */
    bool debug;
    /*! \brief metric */
    std::string metric;
    /*! \brief weight threshold value. 1e-8 */
    double w_minv;

    // declare parameters
    DMLC_DECLARE_PARAMETER(CliParam) {
      DMLC_DECLARE_FIELD(task).set_default("train");
      DMLC_DECLARE_FIELD(model).set_default("ffm");
      DMLC_DECLARE_FIELD(optimizer).set_default("ftrl");
      DMLC_DECLARE_FIELD(sync_mode).set_default("admm");
      DMLC_DECLARE_FIELD(master).set_default("local");
      DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
      DMLC_DECLARE_FIELD(train_path).set_default("");
      DMLC_DECLARE_FIELD(valid_path).set_default("");
      DMLC_DECLARE_FIELD(test_path).set_default("");
      DMLC_DECLARE_FIELD(model_out).set_default("");
      DMLC_DECLARE_FIELD(max_epoch).set_default(10);
      DMLC_DECLARE_FIELD(passes).set_default(1);
      DMLC_DECLARE_FIELD(batch_size).set_default(100);
      DMLC_DECLARE_FIELD(dim).set_default(1e7);
      DMLC_DECLARE_FIELD(field_num).set_default(1);
      DMLC_DECLARE_FIELD(k).set_default(4);
      DMLC_DECLARE_FIELD(nsample_rate).set_default(1);
      DMLC_DECLARE_FIELD(alpha).set_default(0.1);
      DMLC_DECLARE_FIELD(beta).set_default(1.0);
      DMLC_DECLARE_FIELD(l1).set_default(3);
      DMLC_DECLARE_FIELD(l2).set_default(4);
      DMLC_DECLARE_FIELD(debug).set_default(false);
      DMLC_DECLARE_FIELD(metric).set_default("auc");
      DMLC_DECLARE_FIELD(w_minv).set_default(1e-8);
    }
}; // class LearnerTrainingParam
} // namespace mit

#endif // OPENMIT_COMMON_PARAMETER_PARAMETER_H_
