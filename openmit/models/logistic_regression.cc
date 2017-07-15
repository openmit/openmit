#include "openmit/models/logistic_regression.h"

namespace mit {

LR::LR(const mit::KWArgs & kwargs) {
  this->param_.InitAllowUnknown(kwargs);
  if (this->param_.field_num != 0) {
    LOG(INFO) << "WARNING field_num != 0 for linear model. field_num: "
      << this->param_.field_num << ", set to 0.";
    this->param_.field_num = 0;
    LOG(INFO) << "updated field_num: " << this->param_.field_num;
  }
  if (this->param_.k != 0) {
    LOG(INFO) << "WARNING k != 0 for linear model. k: "
      << this->param_.k << ", set to 0.";
    this->param_.k = 0;
  }
  if (!this->param_.is_linear) {
    LOG(INFO) << "WARNING is_linear should be true for lr model, "
      << "this is is_linear = false";
    this->param_.is_linear = true;
  }
}

mit_float LR::Predict(const dmlc::Row<mit_uint> & row,
                      mit::PMAPT & weight, 
                      bool is_norm) {
  mit_float wTx = weight[0]->Get(0);
  for (auto idx = 0u; idx < row.length; ++idx) {
    wTx += weight[row.index[idx]]->Get(0) * row.value[idx];
  }
  if (is_norm) return mit::math::sigmoid(wTx);
  return wTx;
}

mit_float LR::Predict(const dmlc::Row<mit_uint> & row,
                      const mit::SArray<mit_float> & weight,
                      bool is_norm) {
  auto wTx = row.SDot(weight.data(), weight.size());
  if (is_norm) return mit::math::sigmoid(wTx);
  return wTx;
}

void LR::Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::PMAPT & weight,
                  mit::PMAPT * grad) {
  mit_float residual = pred - row.get_label();
  (*grad)[0]->Set(0, residual * 1);   // bias
  for (auto i = 0u; i < row.length; ++i) {    // linear item
    mit_uint feati = row.index[i];
    mit_float partial_wi = residual * row.value[i];
    (*grad)[feati]->Set(0, (*grad)[feati]->Get(0) + partial_wi);
  }
}

void LR::Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  const mit::SArray<mit_float> & weight,
                  mit::SArray<mit_float> * grad) {
  auto residual = pred - row.get_label();
  (*grad)[0] = residual * 1;
  for (auto i = 0u; i < row.length; ++i) {
    (*grad)[row.index[i]] = residual * row.value[i];
  }
}

} // namespace mit
