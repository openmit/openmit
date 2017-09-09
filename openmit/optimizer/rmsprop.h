/*!
 *  Copyright 2016 by Contributors
 *  \file rmsprop.h
 *  \brief the Root Mean Square Propagation (RMSProp) optimization algorithm
 *  \author ZhouYong
 */
#ifndef OPENMIT_OPTIMIZER_RMSPROP_H_
#define OPENMIT_OPTIMIZER_RMSPROP_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {
/*!
 * \brief RMSProp algorithm parameter
 */
class RMSPropParam : public dmlc::Parameter<RMSPropParam> {
  public:
    /*! \brief learning rate initilazation value */
    float lr;
    /*! \brief gamma gradient forgetting factor */
    float gamma;
    /*! \brief epsilon avoid denominator equals to 0 */
    float epsilon;

    /*! \brief declare member */
    DMLC_DECLARE_PARAMETER(RMSPropParam) {
      DMLC_DECLARE_FIELD(lr).set_default(0.1);
      DMLC_DECLARE_FIELD(gamma).set_default(0.99);
      DMLC_DECLARE_FIELD(epsilon).set_default(1e-8);
    }
}; // class RMSPropParam

/*!
 * \brief RMSProp optimization algorithm
 */
class RMSProp : public Opt {
  public:
    /*! \brief default constructor */
    RMSProp(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    ~RMSProp();

    /*! \brief fetch rmsprop optimizer */
    static RMSProp * Get(const mit::KWArgs & kwargs) {
      return new RMSProp(kwargs);
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
    /*! \brief param_ parameter for rmsprop */
    RMSPropParam param_;
    /*! \brief nm_ gradient means square */
    PMAPT nm_;

}; // class RMSProp

DMLC_REGISTER_PARAMETER(RMSPropParam);

RMSProp::RMSProp(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
}

RMSProp::~RMSProp() { }
    
void RMSProp::Update(const dmlc::Row<mit_uint> & row, 
                     mit_float pred, 
                     mit::SArray<mit_float> & weight) {
  // TODO
}

void RMSProp::Update(const mit_uint key, 
                     const uint32_t idx, 
                     const uint32_t size, 
                     const mit_float g, 
                     mit_float & w) {
  if (nm_.find(key) == nm_.end()) {
    nm_.insert(std::make_pair(key, new mit::Unit(size)));
  }
  auto vw = param_.gamma * nm_[key]->Get(idx) + (1 - param_.gamma) * g * g;
  nm_[key]->Set(idx, vw);
  w -= param_.lr / std::sqrt(vw + param_.epsilon) * g;
} // RMSProp::Update

} // namespace mit
#endif // OPENMIT_OPTIMIZER_RMSPROP_H_
