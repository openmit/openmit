/*!
 *  Copyright (c) 2016 by Contributors
 *  \file formula.h
 *  \brief basic match formula, such as sigmoid, distance etc.
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_MATH_BAISC_FORMULA_H_
#define OPENMIT_TOOLS_MATH_BAISC_FORMULA_H_

#include <cmath>

namespace mit {
namespace math {
/*!
 * \brief sigmoid function be used to scenes that normal.:ized to (0,1).
 *        such as generalized linear model.
 */
inline float sigmoid(const float & x) {
  return 1.0 / (1.0 + std::exp(-std::max(std::min(x, 35.0f), -35.0f)));
};

} // namespace math
} // namespace mit
#endif // OPENMIT_TOOLS_MATH_BAISC_FORMULA_H_
