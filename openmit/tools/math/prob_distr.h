/*!
 *  Copyright 2017 by Contributors
 *  \file prob_distr.h
 *  \brief probability distrbution, such as gaussian, uniform ...
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_MATH_PROB_DISTR_H_
#define OPENMIT_TOOLS_MATH_PROB_DISTR_H_

#include <random>
#include "openmit/common/parameter/model_param.h"

namespace mit {
namespace math {
/*!
 * \brief probability distribution struct
 */
struct ProbDistr {
  /*! \brief default generator engine */
  std::default_random_engine gen;

  /*! \brief create a probability distr */
  static ProbDistr * Create(mit::ModelParam & model_param);

  /*! \brief fetch random value */
  virtual float random() = 0;

}; // struct ProbDistr

/*! 
 * \brief normal (gaussian) distribution 
  */
struct NormalDistr : ProbDistr {
  /*! \brief normal distribution */
  std::normal_distribution<float> distr;

  /*! \brief constructor by mean & variance */
  NormalDistr(mit::ModelParam & model_param) {
    distr = std::normal_distribution<float>(
      model_param.random_mean, model_param.random_variance);
  }
  
  /*! \brief get a normal distribution */
  static NormalDistr * Get(mit::ModelParam & model_param) {
    return new NormalDistr(model_param);
  }

  /*! \brief generate random value */
  float random() override { return distr(gen); }

}; // struct NormalDistr

/*! 
 * \brief uniform distribution
 */
struct UniformDistr : ProbDistr {
  /*! \brief uniform distrbution */
  std::uniform_real_distribution<float> distr;

  /*! \brief cosntructor by [-threshold, +threshold] */
  UniformDistr(mit::ModelParam & model_param) {
    distr = std::uniform_real_distribution<float>(
      -model_param.random_threshold, model_param.random_threshold);
  }

  /*! \brief get a uniform distribution */
  static UniformDistr * Get(mit::ModelParam & model_param) {
    return new UniformDistr(model_param);
  }

  /*! \brief generate random value */
  float random() override { return distr(gen); }
  
}; // struct UniformDistr

} // namespace math
} // namespace mit 
#endif // OPENMIT_TOOLS_MATH_PROB_DISTR_H_
