
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
    /*! \brief rho decay factor */
    float rho;
    /*! \brief epsilon used to avoiding denominator equals to 0 */
    float epsilon;

    /*! \brief declare member */
    DMLC_DECLARE_PARAMETER(AdaDeltaParam) {
      DMLC_DECLARE_FIELD(l1).set_default(0.1);
      DMLC_DECLARE_FIELD(l2).set_default(1.0);
      DMLC_DECLARE_FIELD(rho).set_default(0.99);
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

    /*! \brief update for parameter server */
    void Update(PMAPT & grad, PMAPT * weight) override;

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

void AdaDelta::Update(PMAPT & grad, PMAPT * weight) {
  // OpenMP
  for (auto & kunit : grad) {
    auto key = kunit.first;
    mit::Unit * unit = kunit.second;
    auto size = unit->Size();
    CHECK(size >= 1) << "length of unit should not less than 1.";

    if (Eg_.find(key) == Eg_.end()) {
      Eg_.insert(std::make_pair(key, new mit::Unit(size)));
    }

    if (Edx_.find(key) == Edx_.end()) {
      Edx_.insert(std::make_pair(key, new mit::Unit(size)));
    }
    // OpenMP
    for (auto idx = 0u; idx < size; ++idx) {
      auto w = (*weight)[key]->Get(idx);
      auto g = grad[key]->Get(idx);
      // g += param_.l1 * 1 + param_.l2 * w;
      auto eg = param_.rho * Eg_[key]->Get(idx) + (1 - param_.rho) * g * g;
      Eg_[key]->Set(idx, eg);
      auto deltax = - g * RMS(Edx_[key]->Get(idx)) / RMS(Eg_[key]->Get(idx));
      auto edx = param_.rho * Edx_[key]->Get(idx) + (1 - param_.rho) * deltax * deltax;
      Edx_[key]->Set(idx, edx);
      (*weight)[key]->Set(idx, w + deltax);
    }
  }
}


float AdaDelta::RMS(float z) {
  return std::sqrt(z + param_.epsilon);
}

}; // namespace mit

#endif // OPENMIT_OPTIMIZER_ADADELETA_H_
