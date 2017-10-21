//#include "prob_distr.h"
#include "openmit/tools/math/prob_distr.h"

namespace mit {
namespace math {

ProbDistr * ProbDistr::Create(mit::ModelParam & model_param) {
  if (model_param.random_name == "normal" ||
      model_param.random_name == "gaussian") {
    return mit::math::NormalDistr::Get(model_param);
  } else if (model_param.random_name == "uniform") {
    return mit::math::UniformDistr::Get(model_param);
  } else {
    LOG(FATAL) << "it's not support random distribution: " 
      << model_param.random_name 
      << ", and supoort in [normal, gaussian].";
    return nullptr;
  }
}

} // namespace math
} // namespace mit
