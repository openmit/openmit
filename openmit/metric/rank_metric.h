/*!
 * Copyright 2016 by Contributors
 * \file rank_metric.h
 * \brief interface of evalution metric function supported in openmit
 * \author ZhouYong
 */
#ifndef OPENMIT_METRIC_RANK_METRIC_H_
#define OPENMIT_METRIC_RANK_METRIC_H_

#include <algorithm>
#include <cmath>

#include "openmit/metric/metric.h"

namespace mit {
namespace metric {
/*!
 * \brief 
 */
class Auc : public Metric {
  public:
    /*! \brief metric method name */
    const char * Name() const override { return "auc"; }
    /*! \brief get auc method pointer */
    static Auc * Get() { return new Auc(); }
    /*! \brief eval compute */
    inline float Eval(const std::vector<float> & preds,
                      const std::vector<float> & labels) const override;
}; // class Auc

inline float Auc::
Eval(const std::vector<float> & preds,
     const std::vector<float> & labels) const {
  CHECK_EQ(preds.size(), labels.size()) 
    << "preds.size shold be equal to labels.size";
  struct Pair { 
    float pred;
    float label;
  };
  std::vector<Pair> middle(preds.size());
  for (auto i = 0u; i < preds.size(); ++i) {
    middle[i].pred = preds[i];
    middle[i].label = labels[i];
  }
  std::sort(middle.data(), middle.data() + middle.size(), 
      [](const Pair & a, const Pair & b) { return a.pred < b.pred; });
  float auc = 0.0f, cum_tp = 0.0f;
  for (auto i = 0u; i < middle.size(); ++i) {
    if (middle[i].label > 0) {
      cum_tp += 1;
    } else {
      auc += cum_tp;
    } 
  }
  if (fabs(cum_tp - 0) < 1e-10 || fabs(cum_tp - middle.size()) < 1e-10) {
    return 0.0f;
  } else {
    auc /= cum_tp * (middle.size() - cum_tp);
    return auc >= 0.5 ? auc : 1 - auc;
  }
}

} // namespace metric

} // namespace mit

#endif // OPENMIT_METRIC_RANK_METRIC_H_
