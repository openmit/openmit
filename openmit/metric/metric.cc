#include "openmit/metric/metric.h"
#include "openmit/metric/pointwise_metric.h"
#include "openmit/metric/rank_metric.h"

namespace mit {

Metric * Metric::Create(std::string & name) {
  if (name == "logloss") {
    return mit::metric::LogLoss::Get();
  } else if (name == "auc") {
    return mit::metric::Auc::Get();
  } else if (name == "mse") {
    return mit::metric::MSE::Get();
  } else if (name == "rmse") {
    return mit::metric::RMSE::Get();
  } else if (name == "mae") {
    return mit::metric::MAE::Get();
  } else {
    LOG(ERROR) << "metric method not in [logloss, auc, mse, rmse, mae]. " << name; 
    return nullptr;
  }
}

} // namespace mit
