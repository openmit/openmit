/*!
 *  Copyright 2016 by Contributors
 *  \file sgd.h
 *  \brief (stochastic) gradient descent
 *  \author ZhouYong
 */
#ifndef OPENMIT_OPTIMIZER_SGD_H_
#define OPENMIT_OPTIMIZER_SGD_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {
/*!
 * \brief optimizer: gradient descent algorithm
 *        support: sgd/batch-gd
 */
class SGDOptimizer : public Optimizer {
  public:
    /*! \brief constructor for SGD */
    SGDOptimizer(const mit::KWArgs& kwargs);
    
    /*! \brief destructor */
    ~SGDOptimizer();
    
    /*! \brief get SGD optimizer */
    static SGDOptimizer* Get(const mit::KWArgs& kwargs) {
      return new SGDOptimizer(kwargs);
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
}; // class SGD


SGDOptimizer::SGDOptimizer(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  this->param_w_.InitAllowUnknown(kwargs);
  LOG(INFO) << "Learning rate: " << this->param_w_.lr;
}

SGDOptimizer::~SGDOptimizer() {}

void SGDOptimizer::Update(const mit_uint idx, const mit_float g, mit_float & w) {
  w -= param_.lr * g;
}

void SGDOptimizer::Update(const mit::OptimizerParam& param, 
                          const mit_uint& key, 
                          const size_t& idx, 
                          const mit_float& g,
                          mit_float& w, 
                          mit::Entry* weight) {
  w -= param.lr * g;
} // SGDOptimizer::Update

void SGDOptimizer::Update(const mit_uint& key, const size_t& idx, const mit_float& g, mit_float& w, mit::Entry* weight) {
  w -= param_.lr * g;
} 

} // namespace mit
#endif // OPENMIT_OPTIMIZER_SGD_H_
