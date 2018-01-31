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
  private:
    /*! \brief gradient descent parameter */
    mit::OptimizerParam param_; 
}; // class LBFGS


LBFGSOptimizer::LBFGSOptimizer(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  this->param_w_.InitAllowUnknown(kwargs);
}

LBFGSOptimizer::~LBFGSOptimizer() {}

void LBFGSOptimizer::Update(const mit_uint idx, const mit_float g, mit_float & w) {
  w -= param_.lr * g;
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

} // namespace mit
#endif // OPENMIT_OPTIMIZER_LBFGS_H_
