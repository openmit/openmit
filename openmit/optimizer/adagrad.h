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
}

AdaGrad::~AdaGrad() {
  // TODO
}

void AdaGrad::Update(const dmlc::Row<mit_uint> & row, 
                     mit_float pred, 
                     mit::SArray<mit_float> & weight) {
  // TODO
}

void AdaGrad::Update(const mit_uint key, 
                      const uint32_t idx, 
                      const uint32_t size, 
                      const mit_float g, 
                      mit_float & w) {
  if (nm_.find(key) == nm_.end()) {
    nm_.insert(std::make_pair(key, new mit::Unit(size)));
  }
  auto nabla_w = nm_[key]->Get(idx) + g * g;
  nm_[key]->Set(idx, nabla_w);
  auto eta = param_.lrate / std::sqrt(nabla_w + param_.epsilon);
  w -= eta * g;
} // AdaGrad::Update

} // namespace mit
#endif // OPENMIT_OPTIMIZER_ADAGRAD_H_
