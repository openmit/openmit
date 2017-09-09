/*!
 *  Copyright 2016 by Contributors
 *  \file adadelta.h
 *  \brief the AdaDelta optimization algorithm
 *  \author ZhouYong
 */
#ifndef OPENMIT_OPTIMIZER_ADADELETA_H_ 
#define OPENMIT_OPTIMIZER_ADADELETA_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {
/*!
 * \brief adadelta parameter
 */
class AdaDeltaParam : public dmlc::Parameter<AdaDeltaParam> {
  public:
    /*! \brief l1 */
    float l1;
    /*! \brief l2 */
    float l2;
    /*! \brief gamma decay factor */
    float gamma;
    /*! \brief epsilon used to avoiding denominator equals to 0 */
    float epsilon;

    /*! \brief declare member */
    DMLC_DECLARE_PARAMETER(AdaDeltaParam) {
      DMLC_DECLARE_FIELD(l1).set_default(0.1);
      DMLC_DECLARE_FIELD(l2).set_default(1.0);
      DMLC_DECLARE_FIELD(gamma).set_default(0.99);
      DMLC_DECLARE_FIELD(epsilon).set_default(1e-8);
    }
}; // class AdaDeltaParam

/*!
 * \brief adadelta optimization algorithm
 */
class AdaDelta : public Opt {
  public:
    /*! \brief constructor */
    AdaDelta(const mit::KWArgs & kwargs);
    
    /*! \brief destructor */
    ~AdaDelta();

    /*! \brief get a adadelta algorithm */
    static AdaDelta * Create(const mit::KWArgs & kwargs) {
      return new AdaDelta(kwargs);
    }

    /*! \brief update for mpi */
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
    /*! \brief root mean squared */
    float RMS(float z);
  
  private:
    /*! \brief parameter for adadelta */
    AdaDeltaParam param_;
    /*! \brief gradient stored */
    PMAPT Eg_;
    /*! \brief delta x stored */
    PMAPT Edx_;
    
}; // class AdaDelta

DMLC_REGISTER_PARAMETER(AdaDeltaParam);

AdaDelta::AdaDelta(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
}

AdaDelta::~AdaDelta() {
  // TODO  free Eg_ and Edx_
}

void AdaDelta::Update(const dmlc::Row<mit_uint> & row, 
                      mit_float pred, 
                      mit::SArray<mit_float> & weight) {
  // TODO
}

void AdaDelta::Update(const mit_uint key, 
                      const uint32_t idx, 
                      const uint32_t size, 
                      const mit_float g, 
                      mit_float & w) {
  if (Eg_.find(key) == Eg_.end()) {
    Eg_.insert(std::make_pair(key, new mit::Unit(size)));
    Edx_.insert(std::make_pair(key, new mit::Unit(size)));
  }
  mit::Unit * unitEg = Eg_[key];
  mit::Unit * unitEdx = Edx_[key];

  auto eg = param_.gamma * unitEg->Get(idx) + (1 - param_.gamma) * g * g;
  unitEg->Set(idx, eg);
  auto dx = -g * RMS(unitEdx->Get(idx)) / RMS(Eg_[key]->Get(idx)); // deltax
  auto edx = param_.gamma * unitEdx->Get(idx) + (1 - param_.gamma) * dx * dx;
  unitEdx->Set(idx, edx);
  w += dx;
}

float AdaDelta::RMS(float z) {
  return std::sqrt(z + param_.epsilon);
}

}; // namespace mit

#endif // OPENMIT_OPTIMIZER_ADADELETA_H_
