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
/*!
 * \brief logloss metric apply to binary problem
 */
class LogLoss : public Metric {
  public:
    const char* Name() const override {
      return "logloss";
    }
    
    static LogLoss* Get() { return new LogLoss(); }
    
    inline float Eval(const std::vector<float>& preds,
                      const std::vector<float>& labels) const override;
    
    inline float EvalRow(float pred, float y) const;
}; // class LogLoss

class MSE : public Metric {
  public:
    const char* Name() const override {
      return "mse";
    }
    
    static MSE* Get() { return new MSE(); }
    
    inline float Eval(const std::vector<float>& preds,
                      const std::vector<float>& labels) const override;
    
    inline float EvalRow(float pred, float y) const;
}; // class MSE

class RMSE : public Metric {
  public:
    const char* Name() const override {
      return "rmse";
    }
    
    static RMSE* Get() { return new RMSE(); }
    
    inline float Eval(const std::vector<float>& preds,
                      const std::vector<float>& labels) const override;
    
    inline float EvalRow(float pred, float y) const;
}; // class RMSE

class MAE : public Metric {
  public:
    const char* Name() const override {
      return "mae";
    }
    
    static MAE* Get() { return new MAE(); }
    
    inline float Eval(const std::vector<float>& preds,
                      const std::vector<float>& labels) const override;
    
    inline float EvalRow(float pred, float y) const;
}; // class MAE

// implement
inline float LogLoss::Eval(const std::vector<float>& preds, 
                           const std::vector<float>& labels) const {
  auto ndata = labels.size(); CHECK(ndata > 0);
  CHECK_EQ(labels.size(), preds.size()) << "not match size";
  float sum = 0.0;
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for (auto i = 0u; i < ndata; ++i) {
    sum += EvalRow(preds[i], labels[i]);
  }
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

// implement mse(mean squared error) loss
inline float MSE::Eval(const std::vector<float>& preds, 
                       const std::vector<float>& labels) const {
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
inline float MSE::EvalRow(float pred, float y) const {
  return (pred - y) * (pred - y);
}

// implement rmse(root mean squared error) loss
inline float RMSE::Eval(const std::vector<float>& preds, 
                        const std::vector<float>& labels) const {
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
inline float RMSE::EvalRow(float pred, float y) const {
  return (pred - y) * (pred - y);
}

// implement MAELoss(mean absolute error) loss
inline float MAE::Eval(const std::vector<float>& preds, 
                       const std::vector<float>& labels) const {
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
inline float MAE::EvalRow(float pred, float y) const {
  return fabs(pred - y);
}

} // namespace metric
} // namespace mit

#endif // OPENMIT_METRIC_POINTWISE_METRIC_H_
