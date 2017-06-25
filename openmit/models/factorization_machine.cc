#include "openmit/models/factorization_machine.h"

namespace mit {
    
FM::FM(const mit::KWArgs & kwargs) {
  this->param_.InitAllowUnknown(kwargs);
  if (this->param_.field_num != 1) {
    LOG(INFO) << "field_num != 1 for factorization machine model. " 
      << "field_num: " << this->param_.field_num << ", set to 1.";
    this->param_.field_num = 1;
  }
  CHECK(this->param_.k > 0) << "param_.k <= 0 for fm model is error.";
}

mit_float FM::Predict(
    const dmlc::Row<mit_uint> & row,
    const mit::SArray<mit_float> & weight,
    bool is_norm) {
  // TODO
  return 0.0f;
}
    
// implementation of fm prediction based on one instance
mit_float FM::Predict(const dmlc::Row<mit_uint> & row, 
                      std::unordered_map<mit_uint, mit::Unit * > & weight,
                      bool is_norm) {
  CHECK_EQ(param_.field_num, 1);
  auto predict_raw = PredictRaw(row, weight);
  if (is_norm) {
    return mit::math::sigmoid(predict_raw);
  } 
  return predict_raw;
}

mit_float FM::
PredictRaw(const dmlc::Row<mit_uint> & row, 
           std::unordered_map<mit_uint, mit::Unit * > & weight) {
  mit_float linear = 0;
  if (this->param_.is_linear) {
    linear = Linear(row, weight);
  }
  mit_float cross = Cross(row, weight);
  return linear + cross;
}

mit_float FM::
Linear(const dmlc::Row<mit_uint> & row,
       std::unordered_map<mit_uint, mit::Unit * > & weight) {
  mit_float linear = 0;
  linear += weight[0]->Get(0);  // bias w0
  // OpenMP ?
  for (auto i = 0u; i < row.length; ++i) {
    linear += weight[row.index[i]]->Get(0) * row.value[i];
  }
  return linear;
}

mit_float FM::
Cross(const dmlc::Row<mit_uint> & row,
      std::unordered_map<mit_uint, mit::Unit * > & weight) {
  mit_float cross = 0;
  for (auto f = 0u; f < param_.k; ++f) {
    mit_float linear_sum_quad = 0;
    mit_float quad_linear_sum = 0;
    for (auto i = 0u; i < row.length; ++i) {
      auto vif = weight[row.index[i]]->Get(1 + f);
      auto xi = row.value[i];
      linear_sum_quad += vif * xi;
      quad_linear_sum += vif * vif * xi * xi;
    }
    cross += linear_sum_quad * linear_sum_quad - quad_linear_sum;
  }
  return 0.5 * cross;
}

/**
 * for logistic regression:  residual = label - pred;
 * for w0: residual * 1
 * for wi: residual * xi
 * for w(i,f): residual * (xi * \sum_{j=1}^{n} (v(j,f) * xj) - v(i,f) * xi^2)
 */
void FM::Gradient(
    const dmlc::Row<mit_uint> & row, 
    const mit_float & pred,
    std::unordered_map<mit_uint, mit::Unit * > & weight,
    std::unordered_map<mit_uint, mit::Unit * > * grad) {

  mit_float residual = pred - row.get_label();

  if (!this->param_.is_linear) {
    // 0-order bias
    (*grad)[0]->Set(0, residual * 1);

    // 1-order linear item gradient
    // TODO openmp?
    for (auto i = 0u; i < row.length; ++i) {
      mit_uint fid = row.index[i];
      mit_float partial_wi = residual * row.value[i];
      (*grad)[fid]->Set(0, (*grad)[fid]->Get(0) + partial_wi);
    }
  }
  
  // 2-order cross item gradient
  for (auto i = 0u; i < row.length; ++i) {
    auto fi = row.index[i];
    auto xi = row.index[i];
    for (auto f = 0u; f < param_.k; ++f) {
      auto sum = 0.f;
      for (auto j = 0u; j < row.length; ++j) {
        if (i == j) continue;
        auto fj = row.index[j];
        auto xj = row.value[j];
        sum += weight[fj]->Get(1 + f) * xj;
      }
      auto partial_wif = residual * (xi * sum);

      auto updated_grad = (*grad)[fi]->Get(1 + f) + partial_wif;
      (*grad)[fi]->Set(1 + f, updated_grad);
    }
  } // 2-order
} // method FM::Gradient

void FM::Gradient(
    const dmlc::Row<mit_uint> & row,
    const mit_float & pred,
    const mit::SArray<mit_float> & weight,
    mit::SArray<mit_float> * grad) {
  // TODO
}

} // namespace mit 
