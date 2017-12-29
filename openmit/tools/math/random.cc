#include "openmit/tools/math/random.h"
using namespace mit::math;

namespace mit {
namespace math {

Random * Random::Create(mit::ModelParam & model_param) {
  if (model_param.random_name == "normal" ||
      model_param.random_name == "gaussian") {
    return NormalDistr::Get(model_param);
  } else if (model_param.random_name == "uniform") {
    return UniformDistr::Get(model_param);
  } else {
    LOG(FATAL) << "it's not support random distribution: " 
      << model_param.random_name 
      << ", and supoort in [normal, gaussian].";
    return nullptr;
  }
}

} // namespace math
} // namespace mit
