/*!
 * Copyright 2016 by Contributors
 * \file pointwise_metric.h
 * \brief interface of evalution metric function supported in openmit
 * \author ZhouYong,WangYongJie
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

class MSELoss : public Metric {
  public:
    const char * Name() const override {
      return "mse_loss";
    }
    
    static MSELoss * Get() { return new MSELoss(); }
    
    inline float Eval(const std::vector<float> & preds,
                      const std::vector<float> & labels) const override;
    
    inline float EvalRow(float pred, float y) const;

}; // class MSELoss

// implement mse(mean squared error) loss
inline float MSELoss::Eval(const std::vector<float> & preds, const std::vector<float> & labels) const {
  CHECK_NE(labels.size(), 0) << "label cannot be empty!";
  CHECK_NE(preds.size(), 0) << "prediction variable cannot be empty!";
  CHECK_EQ(labels.size(), preds.size()) << "label and prediction size not match, ";
  // TODO omp_ulong ndata = static_cast<omp_ulong>(info.labels.size());
  float sum = 0.0;
  //#pragma omp parallel for reduction(+: sum, wsum) schedule(static)
  auto ndata = labels.size();
  for (auto i = 0u; i < ndata; ++i) {
    sum += EvalRow(preds[i], labels[i]);
  }
  return sum / ndata;
}

// squared error loss implement
inline float MSELoss::EvalRow(float pred, float y) const {
  return (pred - y) * (pred - y);
}

class RMSELoss : public Metric {
  public:
    const char * Name() const override {
      return "rmse_loss";
    }
    
    static RMSELoss * Get() { return new RMSELoss(); }
    
    inline float Eval(const std::vector<float> & preds,
                      const std::vector<float> & labels) const override;
    
    inline float EvalRow(float pred, float y) const;

}; // class RMSELoss

// implement rmse(root mean squared error) loss
inline float RMSELoss::Eval(const std::vector<float> & preds, const std::vector<float> & labels) const {
  CHECK_NE(labels.size(), 0) << "label cannot be empty!";
  CHECK_NE(preds.size(), 0) << "prediction variable cannot be empty!";
  CHECK_EQ(labels.size(), preds.size()) << "label and prediction size not match, ";
  // TODO omp_ulong ndata = static_cast<omp_ulong>(info.labels.size());
  float sum = 0.0;
  //#pragma omp parallel for reduction(+: sum, wsum) schedule(static)
  auto ndata = labels.size();
  for (auto i = 0u; i < ndata; ++i) {
    sum += EvalRow(preds[i], labels[i]);
  }
  return sqrt(sum / ndata);
}

// squared error loss implement
inline float RMSELoss::EvalRow(float pred, float y) const {
  return (pred - y) * (pred - y);
}

class MAELoss : public Metric {
  public:
    const char * Name() const override {
      return "mae_loss";
    }
    
    static MAELoss * Get() { return new MAELoss(); }
    
    inline float Eval(const std::vector<float> & preds,
                      const std::vector<float> & labels) const override;
    
    inline float EvalRow(float pred, float y) const;

}; // class MAELoss

// implement mae(mean absolute error) loss
inline float MAELoss::Eval(const std::vector<float> & preds, const std::vector<float> & labels) const {
  CHECK_NE(labels.size(), 0) << "label cannot be empty!";
  CHECK_NE(preds.size(), 0) << "prediction variable cannot be empty!";
  CHECK_EQ(labels.size(), preds.size()) << "label and prediction size not match, ";
  // TODO omp_ulong ndata = static_cast<omp_ulong>(info.labels.size());
  float sum = 0.0;
  //#pragma omp parallel for reduction(+: sum, wsum) schedule(static)
  auto ndata = labels.size();
  for (auto i = 0u; i < ndata; ++i) {
    sum += EvalRow(preds[i], labels[i]);
  }
  return sum / ndata;
}

// squared error loss implement
inline float MAELoss::EvalRow(float pred, float y) const {
  return fabs(pred - y);
}

} // namespace metric

} // namespace mit

#endif // OPENMIT_METRIC_POINTWISE_METRIC_H_
