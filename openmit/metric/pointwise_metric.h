/*!
 * Copyright 2016 by Contributors
 * \file pointwise_metric.h
 * \brief interface of evalution metric function supported in openmit
 * \author ZhouYong
 */
#ifndef OPENMIT_METRIC_POINTWISE_METRIC_H_
#define OPENMIT_METRIC_POINTWISE_METRIC_H_

#include <cmath>

#include "openmit/metric/metric.h"

namespace mit {

namespace metric {

class LogLoss : public Metric {
  public:
    const char * Name() const override {
      return "logloss";
    }
    
    static LogLoss * Get() { return new LogLoss(); }
    
    inline float Eval(const std::vector<float> & preds,
                      const std::vector<float> & labels) const override;
    
    inline float EvalRow(float pred, float y) const;

}; // class LogLoss

// implement
inline float LogLoss::
Eval(const std::vector<float> & preds, 
     const std::vector<float> & labels) const {
  CHECK_NE(labels.size(), 0) << "label cannot be empty!";
  CHECK_NE(preds.size(), 0) << "prediction cannot be empty!";
  CHECK_EQ(labels.size(), preds.size()) 
    << "label and prediction size not match, ";
  // TODO omp_ulong ndata = static_cast<omp_ulong>(info.labels.size());
  float sum = 0.0;
  //#pragma omp parallel for reduction(+: sum, wsum) schedule(static)
  auto ndata = labels.size();
  for (auto i = 0u; i < ndata; ++i) {
    sum += EvalRow(preds[i], labels[i]);
  }
  return sum / ndata;
}

// logloss implement
inline float LogLoss::EvalRow(float pred, float y) const {
  const float eps = 1e-15f;
  const float pneg = 1.0f - pred;
  if (pred < eps) {
    return -y * std::log(eps) - (1.0f - y) * std::log(1.0f - eps);
  } else if (pneg < eps) {
    return -y * std::log(1.0f - eps) - (1.0f - y) * std::log(eps);
  } else {
    return -y * std::log(pred) - (1.0f - y) * std::log(pneg);
  }
}

} // namespace metric

} // namespace mit

#endif // OPENMIT_METRIC_POINTWISE_METRIC_H_
