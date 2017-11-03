/*!
 *  Copyright 2017 by Contributors
 *  \file optimizer_param.h
 *  \brief optimization algorithm related parameter
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_PARAMETER_OPTIMIZER_PARAM_H_
#define OPENMIT_COMMON_PARAMETER_OPTIMIZER_PARAM_H_

#include <string>
#include "dmlc/parameter.h"
#include "openmit/common/base.h"

namespace mit {
/*! 
 * \brief optimizer related parameter 
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

  /*! \brief declare field */
  DMLC_DECLARE_PARAMETER(OptimizerParam) {
    DMLC_DECLARE_FIELD(optimizer).set_default("sgd");
    DMLC_DECLARE_FIELD(lr).set_default(0.001);
    DMLC_DECLARE_FIELD(l1).set_default(0.01);
    DMLC_DECLARE_FIELD(l2).set_default(0.01);
    DMLC_DECLARE_FIELD(alpha).set_default(0.001);
    DMLC_DECLARE_FIELD(beta).set_default(0.01);
    DMLC_DECLARE_FIELD(beta1).set_default(0.6);
    DMLC_DECLARE_FIELD(beta2).set_default(0.99);
    DMLC_DECLARE_FIELD(gamma).set_default(0.99);
    DMLC_DECLARE_FIELD(epsilon).set_default(1e-8);
  } 
};
} // namespace mit
#endif // OPENMIT_COMMON_PARAMETER_OPTIMIZER_PARAM_H_ 
