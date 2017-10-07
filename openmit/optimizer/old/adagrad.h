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
    float lr;
    /*! \brief epsilon a smaller value avoid denominator equals to 0 */
    float epsilon;

    /*! \brief  */
    DMLC_DECLARE_PARAMETER(AdaGradParam) {
      DMLC_DECLARE_FIELD(l1).set_default(0.1);
      DMLC_DECLARE_FIELD(l2).set_default(1.0);
      DMLC_DECLARE_FIELD(lr).set_default(0.1);
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

    void Init(mit_uint dim) override { 
      nv_.resize(dim+1, 0.0); 
    }
    
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

  private:
    /*! \brief parameter for adagrad optimizer */
    AdaGradParam param_;
    /*! \brief n[i] gradient squared sum for ps */
    PMAPT nm_;
    /*! \brief n[i] gradient squared sum for mpi */
    mit::SArray<mit_float> nv_;
}; // class AdaGrad

DMLC_REGISTER_PARAMETER(AdaGradParam);

AdaGrad::AdaGrad(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
}

AdaGrad::~AdaGrad() {
  // TODO
}

void AdaGrad::Update(const mit_uint idx, const mit_float g, mit_float & w) {
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
  auto eta = param_.lr / std::sqrt(nabla_w + param_.epsilon);
  w -= eta * g;
} // AdaGrad::Update

} // namespace mit
#endif // OPENMIT_OPTIMIZER_ADAGRAD_H_
