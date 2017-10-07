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
 * \brief gradient descent parameter
 */
class SGDParam : public dmlc::Parameter<SGDParam> {
  public:
    /*! \brief optimizer type. gd/adagrad/... */
    std::string optimizer;
    /*! \brief lr learning rate */
    float lr;
    /*! \brief l1 regularation */
    float l1;
    /*! \brief l2 regularation */
    float l2;
  
    /*! \brief declare parameters */
    DMLC_DECLARE_PARAMETER(SGDParam) {
      DMLC_DECLARE_FIELD(optimizer).set_default("gd");
      DMLC_DECLARE_FIELD(lr).set_default(0.01);
      DMLC_DECLARE_FIELD(l1).set_default(0.1);
      DMLC_DECLARE_FIELD(l2).set_default(0.1);
    }

}; // class SGDParam

/*!
 * \brief optimizer: gradient descent algorithm
 *        support: sgd/batch-gd
 */
class SGDOptimizer : public Optimizer {
  public:
    /*! \brief constructor for SGD */
    SGDOptimizer(const mit::KWArgs & kwargs);
    
    /*! \brief destructor */
    ~SGDOptimizer();
    
    /*! \brief get SGD optimizer */
    static SGDOptimizer * Get(const mit::KWArgs & kwargs) {
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
                mit_float & w) override;
    
    /*! 
     * \brief unit updater for parameter server interface
     * \param key model feature id
     * \param idx model unit index
     * \param size model unit max size
     * \param g gradient of unit index that computed by worker node
     * \param w model parameter of unit index
     */
    void Update(const mit_uint key, 
                const uint32_t idx, 
                const uint32_t size, 
                const mit_float g, 
                mit_float & w) override;
    
    void Update(const mit::OptimizerParam & param, 
                const mit_uint & key, 
                const size_t & idx, 
                const mit_float & g,
                mit_float & w,
                mit::Entry * weight = nullptr) override;


  private:
    /*! \brief gradient descent parameter */
    SGDParam param_;   
}; // class SGD

DMLC_REGISTER_PARAMETER(SGDParam);


SGDOptimizer::SGDOptimizer(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
}

SGDOptimizer::~SGDOptimizer() {}

void SGDOptimizer::Update(const mit_uint idx, 
                 const mit_float g, 
                 mit_float & w) {
  w -= param_.lr * g;
}

void SGDOptimizer::Update(const mit_uint key, 
                 const uint32_t idx, 
                 const uint32_t size, 
                 const mit_float g, 
                 mit_float & w) {
  w -= param_.lr * g;
} // SGDOptimizer::Update

void SGDOptimizer::Update(const mit::OptimizerParam & param, 
                          const mit_uint & key, 
                          const size_t & idx, 
                          const mit_float & g,
                          mit_float & w, 
                          mit::Entry * weight) {
  w -= param.lr * g;
} // SGDOptimizer::Update

} // namespace mit
#endif // OPENMIT_OPTIMIZER_SGD_H_
