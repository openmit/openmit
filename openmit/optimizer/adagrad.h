#ifndef OPENMIT_OPTIMIZER_ADAGRAD_H_
#define OPENMIT_OPTIMIZER_ADAGRAD_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {
/*!
 * \brief adaptive gradient descent algorithm
 */
class AdaGradParam : public dmlc::Parameter<AdaGradParam> {
  public:
    float l1;
    float l2;
    DMLC_DECLARE_PARAMETER(AdaGradParam) {
      DMLC_DECLARE_FIELD(l1).set_default(0.1);
      DMLC_DECLARE_FIELD(l2).set_default(1.0);
    }
}; // class AdaGradParam

class AdaGrad : public Opt {
  public:
    AdaGrad(const mit::KWArgs & kwargs);
    ~AdaGrad();

    static AdaGrad * Get(const mit::KWArgs & kwargs) {
      return new AdaGrad(kwargs);
    }
    /*! \brief parameter updater for mpi */
    void Update(
        const dmlc::Row<mit_uint> & row, 
        mit_float pred, 
        mit::SArray<mit_float> & weight_) override;
    /*! \brief updater */
    void Update(std::unordered_map<ps::Key, mit::Unit * > & map_grad,
                std::unordered_map<ps::Key, mit::Unit * > * weight) override;

  private:
    /*! \brief parameter for adagrad optimizer */
    AdaGradParam param_;
    /*! \brief n[i] gradient squared sum */
    std::unordered_map<mit_uint, mit::Unit * > n_;
}; // class AdaGrad
} // namespace mit

#endif // OPENMIT_OPTIMIZER_ADAGRAD_H_
