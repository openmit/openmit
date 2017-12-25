/*!
 * Copyright 2017 by Contributors
 * \file mean_square_metric.h
 * \brief interface of evalution metric function supported in openmit
 * \author ZhouYong, iamhere1
 */
#ifndef OPENMIT_METRIC_MEANSQUARE_METRIC_H_
#define OPENMIT_METRIC_MEANSQUARE_METRIC_H_

#include <cmath>

#include "openmit/metric/metric.h"

namespace mit {

namespace metric {

class MeanSquareLoss : public Metric {
  public:
    const char * Name() const override {
      return "mean_square_loss";
    }
    
    static MeanSquareLoss * Get() { return new MeanSquareLoss(); }
    
    inline float Eval(const std::vector<float> & preds,
                      const std::vector<float> & labels) const override;
    
    inline float EvalRow(float pred, float y) const;

}; // class MeanSquareLoss

// implement square loss
inline float MeanSquareLoss::
Eval(const std::vector<float> & preds, 
     const std::vector<float> & labels) const {
  CHECK_NE(labels.size(), 0) 
      << "label cannot be empty!";
  CHECK_NE(preds.size(), 0) 
      << "prediction variable cannot be empty!";
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

// square loss implement
inline float MeanSquareLoss::EvalRow(float pred, float y) const {
  return (pred - y) * (pred - y);
}

} // namespace metric

} // namespace mit

#endif // OPENMIT_METRIC_POINTWISE_METRIC_H_
