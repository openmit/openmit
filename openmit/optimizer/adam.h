 /*!
 *  Copyright 2016 by Contributors
 *  \file adam.h
 *  \brief the Adaptive Moment estimation (Adam) optimization algorithm
 *  \author ZhouYong
 */
#ifndef OPENMIT_OPTIMIZER_ADAM_H_
#define OPENMIT_OPTIMIZER_ADAM_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {
/*! 
 * \brief adam parameter
 */
class AdamParam : public dmlc::Parameter<AdamParam> {
  public:
    /*! \brief learning rate */
    float lr;
    /*! \brief 1st moment estimation decay factor. default 0.9 */
    float beta1;
    /*! \brief 2nd moment estimation decay factor. default 0.999 */
    float beta2;
    /*! \brief epsilon a small value to avoid denominator equals to 0 */
    float epsilon;

    /*! \brief declare field */
    DMLC_DECLARE_PARAMETER(AdamParam) {
      DMLC_DECLARE_FIELD(lr).set_default(0.001);
      DMLC_DECLARE_FIELD(beta1).set_default(0.9);
      DMLC_DECLARE_FIELD(beta2).set_default(0.999);
      DMLC_DECLARE_FIELD(epsilon).set_default(1e-8);
    }
}; // class AdamParam

/*!
 * \brief adam optimization algorithm
 */
class Adam : public Opt {
  public:
    /*! \brief constructor */
    Adam(const mit::KWArgs & kwargs);
    
    /*! \brief destructor */
    ~Adam();
    
    /*! \brief fetch adam optimizer */
    static Adam * Get(const mit::KWArgs & kwargs) {
      return new Adam(kwargs);
    }

    /*! \brief unit updater for mpi */
    void Update(const dmlc::Row<mit_uint> & row,
                mit_float pred,
                mit::SArray<mit_float> & weight) override;
    /*! 
     * \brief unit updater for parameter server
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
    /*! \brief param_ parameter for adam */
    AdamParam param_;
    /*! \brief wm_ 1st moment information */
    PMAPT wm_;
    /*! \brief vm_ 2nd moment information */
    PMAPT vm_;

}; // class Adam

DMLC_REGISTER_PARAMETER(AdamParam);

Adam::Adam(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
}

Adam::~Adam() {
  // TODO
}

void Adam::Update(const dmlc::Row<mit_uint> & row, 
                  mit_float pred, 
                  mit::SArray<mit_float> & weight) {
  // TODO
}
    
void Adam::Update(const mit_uint key, 
                  const uint32_t idx, 
                  const uint32_t size, 
                  const mit_float g, 
                  mit_float & w) {
  if (wm_.find(key) == wm_.end()) {
    wm_.insert(std::make_pair(key, new mit::Unit(size)));
    vm_.insert(std::make_pair(key, new mit::Unit(size)));
  }
  wm_[key]->Set(idx, param_.beta1 * wm_[key]->Get(idx) + (1 - param_.beta1) * g);
  vm_[key]->Set(idx, param_.beta2 * vm_[key]->Get(idx) + (1 - param_.beta2) * g * g);
  auto lr_hat = param_.lr * std::sqrt(1 - param_.beta2) / (1 - param_.beta1);
  auto epsilon_hat = param_.epsilon * std::sqrt(1 - param_.beta2);
  w -= lr_hat * wm_[key]->Get(idx) / std::sqrt(vm_[key]->Get(idx) + epsilon_hat);
}


} // namespace mit
#endif // OPENMIT_OPTIMIZER_ADAM_H_
