#include "fieldaware_factorization_machine.h"

namespace mit {

mit_float FFM::Predict(
    const dmlc::Row<mit_uint> & row,
    const mit::SArray<mit_float> & weight,
    bool is_norm) {
  // TODO
  return 0.0f;
}
// TODO is_linear? is_sigmoid?
mit_float FFM::Predict(
    const dmlc::Row<mit_uint> & row, 
    std::unordered_map<mit_uint, mit::Unit * > & weight,
    bool is_norm) {
  mit_float predict_raw = PredictRaw(row, weight);
  if (is_norm) {
    return mit::math::sigmoid(predict_raw);
  }
  return predict_raw;
} // method Predict

mit_float FFM::PredictRaw(
    const dmlc::Row<mit_uint> & row, 
    std::unordered_map<mit_uint, mit::Unit * > & weight) {
  mit_float linear = 0;
  if (this->param_.is_linear) {
    linear = Linear(row, weight);
  }
  mit_float cross = Cross(row, weight);
  return linear + cross;
}

mit_float FFM::
Linear(const dmlc::Row<mit_uint> & row,
    std::unordered_map<mit_uint, mit::Unit * > & weight) {
  mit_float linear = 0;
  linear += weight[0]->Get(0);
  // OpenMP ?
  for (auto i = 0u; i < row.length; ++i) {
    linear += weight[row.index[i]]->Get(0) * row.value[i];
  }
  return linear;
}

mit_float FFM::
Cross(const dmlc::Row<mit_uint> & row,
      std::unordered_map<mit_uint, mit::Unit * > & weight) {
  mit_float cross = 0;
  for (auto i = 0u; i < row.length - 1; ++i) {
    auto xi = row.value[i];
    auto feati = row.index[i];
    auto fi = row.field[i];
    for (auto j = i + 1; j < row.length; ++j) {
      auto xj = row.value[j];
      auto featj = row.index[j];
      auto fj = row.field[j];
      auto inner_prod = 0.0f;
      // OpenMP? Eigen?
      for (auto k = 0u; k < param_.k; ++k) {
        auto vifj = weight[feati]->Get(1 + (fj-1)*param_.k + k);
        auto vjfi = weight[featj]->Get(1 + (fi-1)*param_.k + k);
        inner_prod += vifj * vjfi;
      }
      cross += inner_prod * xi * xj;
    }
  }
  return cross;
}

void FFM::Gradient(
    const dmlc::Row<mit_uint> & row, 
    const mit_float & pred,
    std::unordered_map<mit_uint, mit::Unit * > & weight,
    std::unordered_map<mit_uint, mit::Unit * > * grad) {
  
  auto residual = pred - row.get_label();     // for sigmoid
  
  if (this->param_.is_linear) {
    // 0-order bias
    (*grad)[0]->Set(0, residual * 1);
    // 1-order linear 
    for (auto i = 0u; i < row.length; ++i) {
      mit_uint feati = row.index[i];
      mit_float partial_wi = residual * row.value[i];
      (*grad)[feati]->Set(0, (*grad)[feati]->Get(0) + partial_wi);
    }
  }

  // 2-order cross
  for (auto i = 0u; i < row.length - 1; ++i) {
    auto xi = row.value[i];
    auto feati = row.index[i];
    auto fi = row.field[i];
    for (auto j = i + 1; j < row.length; ++j) {
      auto xj = row.value[j];
      auto featj = row.index[j];
      auto fj = row.field[j];
      
      if (fi == fj) continue;   // same field not cross

      for (auto k = 0u; k < param_.k; ++k) {
        auto index_ifjk = 1 + (fj - 1) * param_.field_num + k;
        auto grad_ifjk = residual * (weight[feati]->Get(index_ifjk) * xi * xj);
        auto index_jfik = 1 + (fi - 1) * param_.field_num + k;
        auto grad_jfik = residual * (weight[featj]->Get(index_jfik) * xi * xj);

        (*grad)[feati]->Set(
            index_ifjk, (*grad)[feati]->Get(index_ifjk) + grad_ifjk);
        (*grad)[featj]->Set(
            index_jfik, (*grad)[featj]->Get(index_jfik) + grad_jfik);
      }
    }
  } 
} // method FFM::Gradient

void FFM::Gradient(
    const dmlc::Row<mit_uint> & row,
    const mit_float & pred,
    const mit::SArray<mit_float> & weight,
    mit::SArray<mit_float> * grad) {
  // TODO
}

} // namespace mit
