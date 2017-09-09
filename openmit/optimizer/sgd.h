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
    /*! \brief lrate learning rate */
    float lrate;
    /*! \brief l1 regularation */
    float l1;
    /*! \brief l2 regularation */
    float l2;
  
    /*! \brief declare parameters */
    DMLC_DECLARE_PARAMETER(SGDParam) {
      DMLC_DECLARE_FIELD(optimizer).set_default("gd");
      DMLC_DECLARE_FIELD(lrate).set_default(0.01);
      DMLC_DECLARE_FIELD(l1).set_default(0.1);
      DMLC_DECLARE_FIELD(l2).set_default(0.1);
    }

}; // class SGDParam

/*!
 * \brief optimizer: gradient descent algorithm
 *        support: sgd/batch-gd
 */
class SGD : public Opt {
  public:
    /*! \brief constructor for SGD */
    SGD(const mit::KWArgs & kwargs);
    
    /*! \brief destructor */
    ~SGD();
    
    /*! \brief get SGD optimizer */
    static SGD * Get(const mit::KWArgs & kwargs) {
      return new SGD(kwargs);
    }
    
    /*! \brief parameter updater for mpi */
    void Update(const dmlc::Row<mit_uint> & row, 
                mit_float pred, 
                mit::SArray<mit_float> & weight_) override;

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

  private:
    /*! \brief gradient descent parameter */
    SGDParam param_;   
}; // class SGD

DMLC_REGISTER_PARAMETER(SGDParam);


SGD::SGD(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
}

SGD::~SGD() {}

void SGD::Update(const dmlc::Row<mit_uint> & row, 
                 mit_float pred, 
                 mit::SArray<mit_float> & weight_) {
  // TODO
}

void SGD::Update(const mit_uint key, 
                 const uint32_t idx, 
                 const uint32_t size, 
                 const mit_float g, 
                 mit_float & w) {
  w -= param_.lrate * g;
} // SGD::Update

} // namespace mit
#endif // OPENMIT_OPTIMIZER_SGD_H_
