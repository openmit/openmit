#include "openmit/models/lr.h"

namespace mit {

void LR::Pull(ps::KVPairs<mit_float>& response, 
              mit::entry_map_type* weight) {
  size_t keys_size = response.keys.size();
  response.vals.resize(keys_size, 0.0f);;
  response.lens.resize(keys_size, 1);

  omp_set_num_threads(cli_param_.num_thread);
  int blocksize = keys_size / cli_param_.num_thread;
  if (keys_size % cli_param_.num_thread != 0) blocksize += 1;
  #pragma omp parallel for schedule(static, blocksize)
  for (auto i = 0u; i < keys_size; ++i) {
    ps::Key key = response.keys[i];
    if (weight->find(key) == weight->end()) {
      #pragma omp critical
      weight->insert(std::make_pair(key, mit::Entry::Create(
        model_param_, entry_meta_.get(), random_.get())));
    }
    response.vals[i] = (*weight)[key]->Get();
  }
}

mit_float LR::Predict(const dmlc::Row<mit_uint> & row, 
                      const std::vector<mit_float> & weights, 
                      mit::key2offset_type & key2offset, 
                      bool is_norm) {
  mit_float wTx = 0;
  for (auto idx = 0u; idx < row.length; ++idx) {
    mit_uint key = row.get_index(idx);
    if (key2offset.find(key) == key2offset.end()) {
      LOG(FATAL) << key << " not in key2offset";
    }
    auto offset = key2offset[key].first;
    wTx += weights[offset] * row.get_value(idx);
  }
  // intercept
  if (! cli_param_.is_contain_intercept) {
    wTx += weights[key2offset[0].first]; 
  }
  if (is_norm) return mit::math::sigmoid(wTx);
  return wTx;
}

void LR::Gradient(const dmlc::Row<mit_uint> & row, 
                  const std::vector<mit_float> & weights, 
                  mit::key2offset_type & key2offset, 
                  std::vector<mit_float> * grads, 
                  const mit_float & lossgrad_value) {
  auto instweight = row.get_weight();
  // TODO OpenMP
  for (auto idx = 0u; idx < row.length; ++idx) {
    auto key = row.get_index(idx);
    auto offset = key2offset[key].first;
    auto xi = row.get_value(idx);
    (*grads)[offset] += lossgrad_value * xi * instweight;
  }
  if (! cli_param_.is_contain_intercept) {
    auto index0 = key2offset[0].first;
    (*grads)[index0] += lossgrad_value * 1 * instweight;
  }
}

void LR::Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::SArray<mit_float> * grad) {
  auto residual = pred - row.get_label();
  auto instweight = row.get_weight();
  (*grad)[0] += residual * 1 * instweight;
  for (auto i = 0u; i < row.length; ++i) {
    auto fvalue = row.get_value(i);
    (*grad)[row.index[i]] += residual * fvalue * instweight;
  }
}

mit_float LR::Predict(const dmlc::Row<mit_uint> & row,
                      const mit::SArray<mit_float> & weight,
                      bool is_norm) {
  auto wTx = row.SDot(weight.data(), weight.size());
  if (is_norm) return mit::math::sigmoid(wTx);
  return wTx;
}

} // namespace mit
