#include "openmit/model/lr.h"

namespace mit {
LR::~LR() {}

LR* LR::Get(const mit::KWArgs& kwargs) {
  return new LR(kwargs);
}

void LR::Pull(ps::KVPairs<mit_float>& response, 
                mit::entry_map_type* weight) {
  size_t keys_size = response.keys.size();
  response.vals.resize(keys_size, 0.0f);;
  response.lens.resize(keys_size, 1);

  auto nthread = cli_param_.num_thread;
  omp_set_num_threads(nthread);
  int chunksize = keys_size / nthread;
  if (keys_size % nthread != 0) chunksize += 1;
  #pragma omp parallel for schedule(static, chunksize)
  for (auto i = 0u; i < keys_size; ++i) {
    ps::Key key = response.keys[i];
    mit::Entry* entry = nullptr;
    if (weight->find(key) == weight->end()) {
      entry = mit::Entry::Create(model_param_, entry_meta_.get(), random_.get());
      CHECK_NOTNULL(entry);
      #pragma omp critical
      {
        std::lock_guard<std::mutex> lk(mu_);
        weight->insert(std::make_pair(key, entry));
      }
    } else {
      entry = (*weight)[key];
    }
    CHECK_NOTNULL(entry); CHECK_EQ(entry->Size(), 1);
    response.vals[i] = entry->Get();
  }
}

mit_float LR::Predict(const dmlc::Row<mit_uint>& row, 
                        const std::vector<mit_float>& weights, 
                        mit::key2offset_type& key2offset) {
  mit_float wTx = 0.0f;
  // intercept 
  auto offset0 = key2offset[0].first;
  if (! cli_param_.is_contain_intercept) {
    wTx += weights[offset0]; 
  }
  #pragma omp parallel for reduction(+:wTx) num_threads(cli_param_.num_thread)
  for (auto idx = 0u; idx < row.length; ++idx) {
    auto key = row.get_index(idx);
    if (! cli_param_.is_contain_intercept && key == 0) continue;
    CHECK(key2offset.find(key) != key2offset.end());
    auto offseti = key2offset[key].first;
    wTx += weights[offseti] * row.get_value(idx);
  }
  return wTx;
} // LR::Predict

void LR::Gradient(const dmlc::Row<mit_uint>& row, 
                    const std::vector<mit_float>& weights, 
                    mit::key2offset_type& key2offset, 
                    std::vector<mit_float>* grads, 
                    const mit_float& loss_grad) {
  auto instweight = row.get_weight();
  auto middle = loss_grad * instweight;
  #pragma omp parallel for num_threads(cli_param_.num_thread)
  for (auto idx = 0u; idx < row.length; ++idx) {
    auto key = row.get_index(idx);
    auto offset = key2offset[key].first;
    (*grads)[offset] += row.get_value(idx) * middle;
  }
  if (! cli_param_.is_contain_intercept) {
    auto offset0 = key2offset[0].first;
    (*grads)[offset0] += 1 * middle;
  }
} // LR::Gradient 

} // namespace mit
