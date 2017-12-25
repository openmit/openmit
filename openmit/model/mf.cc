#include "openmit/model/mf.h"
#include <cmath>

namespace mit {

void MF::Update(const ps::SArray<mit_uint> & keys,
                const ps::SArray<mit_float> & vals,
                const ps::SArray<int> & lens,
                mit::entry_map_type * weight) {
  auto entry_size = model_param_.embedding_size;
  auto keys_length = keys.size();
  auto offset = 0u;
  for (auto i = 0u; i < keys_length; ++i) {
    auto key = keys[i];
    CHECK(weight->find(key) != weight->end())
      << key << " not in model structure";
    CHECK_EQ(lens[i], entry_size)
      << "lens[i] != embedding_size for fm model";
    for (int k = 0; k < lens[i]; ++k) {
      auto v = (*weight)[key]->Get(k);
      auto g = vals[offset++];
      optimizer_v_->Update(key, k, g, v, (*weight)[key]);
      (*weight)[key]->Set(k, v);
    }
  }
  CHECK_EQ(offset, vals.size()) << "offset not match vals.size for model update";
}

void MF::Pull(ps::KVPairs<mit_float> & response,
              mit::entry_map_type * weight) {
  size_t entry_size = model_param_.embedding_size;
  size_t keys_size = response.keys.size();
  size_t vals_size = keys_size * entry_size;
  response.vals.resize(vals_size);
  response.lens.resize(keys_size, entry_size);

  // feature
  omp_set_num_threads(cli_param_.num_thread);
  size_t blocksize = keys_size / cli_param_.num_thread;
  if (keys_size % cli_param_.num_thread != 0) blocksize += 1;
  #pragma omp parallel for schedule(static, blocksize)
  for (auto i = 0u; i < keys_size; ++i) {
    ps::Key key = response.keys[i];
    if (weight->find(key) == weight->end()) {
      mit::Entry * entry = mit::Entry::Create(
        model_param_, entry_meta_.get(), random_.get());
      #pragma omp critical
      weight->insert(std::make_pair(key, entry));
    }
    mit::Entry * entry = (*weight)[key];
    for (auto idx = 0u; idx < entry_size; ++idx) {
      auto index = i * entry_size + idx;
      response.vals[index] = entry->Get(idx);
    }
  }
}

mit_float MF::Predict(const std::vector<mit_float> & user_weights,
                      const size_t user_offset,
                      const std::vector<mit_float> & item_weights,
                      size_t item_offset,
                      size_t factor_len){
  mit_float sum = 0.0f;
  for (size_t i = 0; i < factor_len; i++){
    sum += user_weights[user_offset + i] * item_weights[item_offset + i];
  }
  return sum;
}

void MF::Gradient(const mit_float lossgrad_value,
                  const std::vector<mit_float> & user_weights,
                  const size_t user_offset,
                  const std::vector<mit_float> & item_weights,
                  const size_t item_offset,
                  const mit_uint factor_len,
                  std::vector<mit_float> * user_grads,
                  std::vector<mit_float> * item_grads) {
  //LOG(INFO) << "factor_len:" << factor_len;
  for (size_t k = 0; k < factor_len; k++) {
    //LOG(INFO) << "k:" << k << " (*user_grads)[user_offset + k]:" << (*user_grads)[user_offset + k] << " item_weights[item_offset + k]:" << item_weights[item_offset + k];
    (*user_grads)[user_offset + k] += lossgrad_value * item_weights[item_offset + k];
    //LOG(INFO) << "k:" << k << " (*user_grads)[user_offset + k]:" << (*user_grads)[user_offset + k];
    //LOG(INFO) << "(*item_grads)[item_offset + k]:" << (*item_grads)[item_offset + k] << " user_weights[user_offset + k]:" << user_weights[user_offset + k];
    (*item_grads)[item_offset + k] += lossgrad_value * user_weights[user_offset + k];
    //LOG(INFO) << "k:" << k << " (*item_grads)[item_offset + k]:" << (*item_grads)[item_offset + k];
    //LOG(INFO) << "item_offset+k:" << item_offset + k;
  }
}

} // namespace mit 
