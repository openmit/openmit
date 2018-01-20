#ifndef OPENMIT_LOSS_LOSS_H_
#define OPENMIT_LOSS_LOSS_H_ 

#include <cmath>
#include <functional>
#include <string>
#include "dmlc/logging.h"
#include "openmit/common/base.h"

namespace mit {
/*! \brief loss function type */
typedef std::function<mit_float(const mit_float &, const mit_float &)> lossfunc_type;

/*!
 * \brief loss function 
 */
struct Loss {
  /*! brief create loss object */
  static Loss* Create(std::string type);

  /*! 
   * \brief constructor by register loss expr & gradient 
   */
  Loss(const lossfunc_type& loss, const lossfunc_type& gradient) : 
    loss(loss), gradient(gradient) {} 

  /*! \brief loss function expression */
  lossfunc_type loss;

  /*! \brief gradient logic of loss */
  lossfunc_type gradient;
}; // struct Loss

/*!
 * \brief squared loss 
 */
struct SquaredLoss {
  /*!
   * \brief function of squared loss 
   */
  static mit_float LossFunc(const mit_float& target, 
                            const mit_float& mfunc) {
    return 0.5 * (mfunc - target) * (mfunc - target);
  }

  /*! 
   * \brief gradient or sub-gradients 
   * \param label objective value. category / real value 
   * \param mfunc model function expression
   */
  static mit_float Gradient(const mit_float& target, 
                            const mit_float& mfunc) {
    return mfunc - target;
  }
}; // struct SquaredLoss 

/*!
 * \brief logistic loss 
 */
struct LogitLoss {
  /*!
   * \brief function of squared loss 
   */
  static mit_float LossFunc(const mit_float& label, 
                            const mit_float& mfunc) { 
    auto y = label > 0 ? 1 : -1;
    return std::log(1 + std::exp(-y * mfunc));
  }

  /*! 
   * \brief gradient or sub-gradients 
   * \param label it belongs to {+1, -1}
   * \param mfunc model function expression
   */
  static mit_float Gradient(const mit_float& label, 
                            const mit_float& mfunc) {
    mit_float y = label > 0 ? 1 : -1;
    return -y / (1 + std::exp(y * mfunc));
  }
}; // struct LogitLoss

} // namespace mit 
#endif // OPENMIT_LOSS_LOSS_H_
