#ifndef OPENMIT_OPTIMIZER_ADAGRAD_H_
#define OPENMIT_OPTIMIZER_ADAGRAD_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {
/*!
 * \brief adaptive gradient descent algorithm
 */
class AdaGradParam : public dmlc::Parameter<AdaGradParam> {
  public:
    /*! \brief l1 L1 regularization coefficient */
    float l1;
    /*! \brief l2 L2 regularization coefficient */
    float l2;
    /*! \brief learning rate initilazation value */
    float lrate;
    /*! \brief epsilon for adagrad */
    float epsilon;

    /*! \brief  */
    DMLC_DECLARE_PARAMETER(AdaGradParam) {
      DMLC_DECLARE_FIELD(l1).set_default(0.1);
      DMLC_DECLARE_FIELD(l2).set_default(1.0);
      DMLC_DECLARE_FIELD(lrate).set_default(0.1);
      DMLC_DECLARE_FIELD(epsilon).set_default(1e-8);
    }
}; // class AdaGradParam

class AdaGrad : public Opt {
  public:
    /*! \brief constructor */
    AdaGrad(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    ~AdaGrad();

    static AdaGrad * Get(const mit::KWArgs & kwargs) {
      return new AdaGrad(kwargs);
    }

    /*! \brief updater for mpi */
    void Update(
        const dmlc::Row<mit_uint> & row, 
        mit_float pred, 
        mit::SArray<mit_float> & weight_) override;

    /*! \brief updater for parameter server */
    void Update(PMAPT & map_grad, PMAPT * weight) override;

  private:
    /*! \brief parameter for adagrad optimizer */
    AdaGradParam param_;
    /*! \brief n[i] gradient squared sum for ps */
    PMAPT nm_;
    /*! \brief n[i] gradient squared sum for mpi */
    PVECT nv_;
}; // class AdaGrad

DMLC_REGISTER_PARAMETER(AdaGradParam);

AdaGrad::AdaGrad(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  // TODO
}

AdaGrad::~AdaGrad() {
  // TODO
}

void AdaGrad::Update(
    const dmlc::Row<mit_uint> & row, 
    mit_float pred, 
    mit::SArray<mit_float> & weight) {
  // TODO
}

void AdaGrad::Update(PMAPT & map_grad, PMAPT * weight) {
  // OpenMP 
  for (auto & kunit : map_grad) {
    auto feati = kunit.first;
    mit::Unit * unit = kunit.second;
    auto size = unit->Size();
    CHECK(size >= 1) << "length of unit should not less than 1.";

    if (nm_.find(feati) == nm_.end()) {
      nm_.insert(std::make_pair(feati, new mit::Unit(size)));
    }

    for (auto idx = 0u; idx < size; ++idx) {
      auto w = (*weight)[feati]->Get(idx);
      auto g = map_grad[feati]->Get(idx);
      // auto nabla_w = g + param_.l1 * 1 + param_.l2 * w;
      auto nabla_w = g;
      auto n_w = nm_[feati]->Get(idx) + g * g;
      nm_[feati]->Set(idx, n_w);
      auto eta = param_.lrate / std::sqrt(n_w + param_.epsilon);
      (*weight)[feati]->Set(idx, w - eta * nabla_w);
    }

    LOG(INFO) << "nm_ info feati: " << feati << ",\t unit: " << nm_[feati]->Str();
  }
}

} // namespace mit

#endif // OPENMIT_OPTIMIZER_ADAGRAD_H_
