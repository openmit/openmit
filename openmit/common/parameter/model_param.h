/*!
 *  Copyright 2017 by Contributors
 *  \file model_param.h
 *  \brief model related parameter
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_PARAMETER_MODEL_PARAM_H_
#define OPENMIT_COMMON_PARAMETER_MODEL_PARAM_H_

#include <string>
#include "dmlc/parameter.h"
#include "openmit/common/base.h"

namespace mit {
/*! \brief model related parameter */
struct ModelParam : public dmlc::Parameter<ModelParam> {
  /*! \brief model name */
  std::string model;
  /*! \brief max feature dimension */
  uint32_t max_dim;
  /*! \brief latent vector length for fm/ffm */
  size_t embedding_size;
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
  /*! \brief data format */
  std::string data_format;

  // declare parameters 
  DMLC_DECLARE_PARAMETER(ModelParam) {
    DMLC_DECLARE_FIELD(model).set_default("lr");
    DMLC_DECLARE_FIELD(max_dim).set_default(1e8);
    DMLC_DECLARE_FIELD(embedding_size).set_default(4);
    DMLC_DECLARE_FIELD(field_combine_set).set_default("");
    DMLC_DECLARE_FIELD(field_combine_pair).set_default("");
    DMLC_DECLARE_FIELD(nbit).set_default(5);
      
    DMLC_DECLARE_FIELD(random_name).set_default("normal");
    DMLC_DECLARE_FIELD(random_mean).set_default(0.0);
    DMLC_DECLARE_FIELD(random_variance).set_default(0.01);
    DMLC_DECLARE_FIELD(random_threshold).set_default(0.01);
    
    DMLC_DECLARE_FIELD(data_format).set_default("auto");
  }
}; // struct ModelParam

} // namespace mit_
#endif // OPENMIT_COMMON_PARAMETER_MODEL_PARAM_H_
