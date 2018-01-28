#include "openmit/metric/metric.h"
#include "openmit/metric/pointwise_metric.h"
#include "openmit/metric/rank_metric.h"

namespace mit {

Metric * Metric::Create(std::string & name) {
  if (name == "logloss") {
    return mit::metric::LogLoss::Get();
  } else if (name == "auc") {
    return mit::metric::Auc::Get();
  } else if (name == "mseloss") {
    return mit::metric::MSELoss::Get();
  } else if (name == "rmseloss") {
    return mit::metric::RMSELoss::Get();
  } else if (name == "maeloss") {
    return mit::metric::MAELoss::Get();
  } else {
    LOG(ERROR) << "metric method not in [logloss, auc, mseloss, rmseloss, maeloss]. " << name; 
    return nullptr;
  }
}

} // namespace mit
