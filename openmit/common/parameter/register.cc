#include "openmit/common/parameter/admm_param.h"
#include "openmit/common/parameter/cli_param.h"
#include "openmit/common/parameter/model_param.h"
#include "openmit/common/parameter/optimizer_param.h"

namespace mit {
// register admm parameter
DMLC_REGISTER_PARAMETER(AdmmParam);
// register client parameter
DMLC_REGISTER_PARAMETER(CliParam);
// register model parameter 
DMLC_REGISTER_PARAMETER(ModelParam);
// register optimizer parameter 
DMLC_REGISTER_PARAMETER(OptimizerParam);
} // namespace mit 
