/*!
 *  Copyright 2016 by Contributors
 *  \file admm_param.h
 *  \brief admm algorithm framework configure parameter
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_PARAMETER_ADMM_PARAM_H_
#define OPENMIT_COMMON_PARAMETER_ADMM_PARAM_H_

#include "dmlc/parameter.h"
#include "openmit/common/base.h"

namespace mit {
/*!
 * \brief admm parameter with distributed (decentralized) algorithm
 */
class AdmmParam : public dmlc::Parameter<AdmmParam> {
  public:
    /*! \brief lambda objective function l1-norm parameter */
    float lambda_obj;
    /*!
     * \brief argument lagrangians factor that
     *        used to step size when dual variance updating.
     */
    float rho;
    /*! \brief dim size of global_weights. it equals to feature dimimension. */
    uint64_t dim;
    /*! declare parameter field */
    DMLC_DECLARE_PARAMETER(AdmmParam) {
      DMLC_DECLARE_FIELD(lambda_obj).set_default(0.05);
      DMLC_DECLARE_FIELD(rho).set_default(1);
      DMLC_DECLARE_FIELD(dim).set_default(1e8);
    }
};  // class AdmmParam

} // namespace mit

#endif // OPENMIT_COMMON_PARAMETER_ADMM_PARAM_H_
