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
inline float LogLoss::Eval(const std::vector<float>& preds, const std::vector<float>& labels) const {
  auto ndata = labels.size(); CHECK(ndata > 0);
  CHECK_EQ(labels.size(), preds.size()) << "not match size";
  float sum = 0.0;
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for (auto i = 0u; i < ndata; ++i) {
    sum += EvalRow(preds[i], labels[i]);
  }
  /*
  for (auto i = 0u; i < ndata; ++i) {
    LOG(INFO) << "<p, y>: <" << preds[i] << ", " << labels[i] << ">";
  }
  LOG(INFO) << "LogLoss::Eval sum: " << sum << ", ndata: " << ndata << ", avg(sum): " << sum/ndata;
  */
  return sum / ndata;
}

// logloss implement
inline float LogLoss::EvalRow(float pred, float y) const {
  const float eps = 1e-15f;
  const float pneg = 1.0f - pred;
  float res = 0.0f;
  if (pred < eps) {
    res = -y * std::log(eps) - (1.0f - y) * std::log(1.0f - eps);
  } else if (pneg < eps) {
    res = -y * std::log(1.0f - eps) - (1.0f - y) * std::log(eps);
  } else {
    res = -y * std::log(pred) - (1.0f - y) * std::log(pneg);
  }
  return res;
}

} // namespace metric

} // namespace mit

#endif // OPENMIT_METRIC_POINTWISE_METRIC_H_
