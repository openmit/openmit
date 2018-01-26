#include "openmit/metric/metric.h"
#include "openmit/metric/pointwise_metric.h"
#include "openmit/metric/rank_metric.h"
#include "openmit/metric/meansquare_metric.h"

namespace mit {

Metric * Metric::Create(std::string & name) {
  if (name == "logloss") {
    return mit::metric::LogLoss::Get();
  } else if (name == "auc") {
    return mit::metric::Auc::Get();
  } else if (name == "mean_square_loss") {
    return mit::metric::MeanSquareLoss::Get();
  } else {
    LOG(ERROR) << "metric method not in [logloss, auc]. " << name; 
    return nullptr;
  }
}

} // namespace mit
