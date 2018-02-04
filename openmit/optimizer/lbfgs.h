/*!
 *  Copyright 2018 by Contributors
 *  \file lbfgs.h
 *  \brief LBFGS (limited-memory BFGS or limited-strorate BFGS) optimizer
 *  \author WangYongJie
 */
#ifndef OPENMIT_OPTIMIZER_LBFGS_H_
#define OPENMIT_OPTIMIZER_LBFGS_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {
/*!
 * \brief optimizer: lbfgs algorithm
 */
class LBFGSOptimizer : public Optimizer {
  public:
    /*! \brief constructor for LBFGS */
    LBFGSOptimizer(const mit::KWArgs& kwargs);
    
    /*! \brief destructor */
    ~LBFGSOptimizer();
    
    /*! \brief get LBFGS optimizer */
    static LBFGSOptimizer* Get(const mit::KWArgs& kwargs) {
      return new LBFGSOptimizer(kwargs);
    }
    
    void Init(mit_uint dim) override {}

    /*! \brief initialize the lbfgs parameter*/
    void LBFGSParamInit();

    /*!
     * \brief parameter updater for mpi
     * \param idx model index 
     * \param g gradient of model index 
     * \param w model index weight
     */
     void Update(const mit_uint idx,
                 const mit_float g,
                 mit_float& w) override;

    /*! 
     * \brief model updater for parameter server interface
     * \param param optimizer parameter
     * \param key model feature id
     * \param idx entry data index
     * \param g gradient of unit index that computed by worker node
     * \param w model parameter of unit index 
     * \param weight used initialize optimizer middle variable
     */
    void Update(const mit::OptimizerParam & param, 
                const mit_uint & key, 
                const size_t & idx, 
                const mit_float & g,
                mit_float & w,
                mit::Entry * weight = nullptr) override;
    
    void Update(const mit_uint & key, 
                const size_t & idx, 
                const mit_float & g, 
                mit_float & w, 
                mit::Entry * weight = nullptr) override;

    void Run(int n,
             lbfgsfloatval_t *x,
             lbfgsfloatval_t *fx,
             lbfgs_evaluate_t proc_evaluate,
             lbfgs_progress_t proc_progress,
             void *instance);

  private:
    /*! \brief lbfgs parameter */
    lbfgs_parameter_t lbfgs_param_;
}; // class LBFGS


LBFGSOptimizer::LBFGSOptimizer(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  /*! \brief lbfgs parameter initialization*/
  LBFGSParamInit();
}

void LBFGSOptimizer::LBFGSParamInit() {
    /*! \brief lbfgs parameter initialization*/
  lbfgs_param_ = {
    6, 1e-5, 0, 1e-5,
    0, LBFGS_LINESEARCH_DEFAULT, 40,
    1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
    0.0, 0, -1,
  };
  lbfgs_param_.m = param_.m;
  lbfgs_param_.max_iterations = param_.max_iterations;
  lbfgs_param_.linesearch = param_.linesearch;
  lbfgs_param_.max_linesearch = param_.max_linesearch;
  lbfgs_param_.orthantwise_c = param_.l1;
}

LBFGSOptimizer::~LBFGSOptimizer() {}

void LBFGSOptimizer::Update(const mit_uint idx, const mit_float g, mit_float & w) {
  w = g;
}

void LBFGSOptimizer::Update(const mit::OptimizerParam& param, 
                          const mit_uint& key, 
                          const size_t& idx, 
                          const mit_float& g,
                          mit_float& w, 
                          mit::Entry* weight) {
  w = g;
} // LBFGSOptimizer::Update

void LBFGSOptimizer::Update(const mit_uint& key, const size_t& idx, const mit_float& g, mit_float& w, mit::Entry* weight) {
  w = g;
} 

void LBFGSOptimizer::Run(int n,
                         lbfgsfloatval_t *x,
                         lbfgsfloatval_t *fx,
                         lbfgs_evaluate_t proc_evaluate,
                         lbfgs_progress_t proc_progress,
                         void *instance)
{
  CHECK_NOTNULL(x);
  CHECK_NOTNULL(fx);
  int ret = lbfgs(n, x, fx, proc_evaluate, proc_progress, instance, &lbfgs_param_);
  LOG(INFO) << "L-BFGS optimization terminated with status code = " << ret;
} 

} // namespace mit
#endif // OPENMIT_OPTIMIZER_LBFGS_H_
