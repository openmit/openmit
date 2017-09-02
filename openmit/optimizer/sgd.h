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

    /*! \brief parameter updater for ps */
    void Update(PMAPT & map_grad, PMAPT * weight) override;

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

// w = w - lrate * (grad(loss) + l1 * grad(l1) + l2 * grad(l2))
void SGD::Update(PMAPT & map_grad, PMAPT * weight) {
  // OpenMP
  for (auto & kunit : map_grad) {
    auto feati = kunit.first;
    mit::Unit * unit = kunit.second;
    auto size = unit->Size();
    CHECK(size >= 1) << "length of unit < 1";
    
    for (auto idx = 0u; idx < size; ++idx) {
      auto w = (*weight)[feati]->Get(idx);
      auto g = map_grad[feati]->Get(idx);
      //auto nabla_w = g + param_.l1 * 1 + param_.l2 * w;
      auto nabla_w = g;
      auto updated_w = w - param_.lrate * nabla_w;
      (*weight)[feati]->Set(idx, updated_w);
    }
  }
}

} // namespace mit

#endif // OPENMIT_OPTIMIZER_SGD_H_
