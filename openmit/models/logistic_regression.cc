#include "openmit/models/logistic_regression.h"

namespace mit {

void LR::InitOptimizer(const mit::KWArgs & kwargs) {
  optimizer_.reset(mit::Optimizer::Create(kwargs));
}

void LR::Pull(ps::KVPairs<mit_float> & response, 
              mit::entry_map_type * weight) {
  for (auto i = 0u; i < response.keys.size(); ++i) {
    ps::Key key = response.keys[i];
    if (weight->find(key) == weight->end()) {
      mit::Entry * entry = mit::Entry::Create(
        model_param_, entry_meta_.get(), random_.get());
      weight->insert(std::make_pair(key, entry));
    }
    mit::Entry * entry = (*weight)[key];
    ps::SArray<mit_float> wv;
    wv.CopyFrom(entry->Data(), entry->Size());
    // fill response.vals and response.lens 
    response.vals.append(wv);
    response.lens.push_back(entry->Size());
  }
}

void LR::Update(const ps::SArray<mit_uint> & keys, 
                const ps::SArray<mit_float> & vals, 
                const ps::SArray<int> & lens, 
                mit::entry_map_type * weight) {
  auto keys_length = keys.size();
  auto offset = 0u;
  for (auto i = 0u; i < keys_length; ++i) {
    CHECK_EQ(lens[i], 1) 
      << "length of vals[" << i << "] should be 1 for lr model.";
    if (weight->find(keys[i]) == weight->end()) {
      LOG(FATAL) << keys[i] << " not in weight structure";
    }
    auto w = (*weight)[keys[i]]->Get(0);    // 0: index of w
    auto g = vals[offset];
    optimizer_->Update(keys[i], 0, g, w, (*weight)[keys[i]]);
    (*weight)[keys[i]]->Set(0, w);
    offset++;
  }
}

mit_float LR::Predict(const dmlc::Row<mit_uint> & row, 
                      const std::vector<mit_float> & weights, 
                      std::unordered_map<mit_uint, 
                      std::pair<size_t, int> > & key2offset, 
                      bool is_norm) {
  mit_float wTx = 0;
  for (auto idx = 0u; idx < row.length; ++idx) {
    mit_uint featid = row.get_index(idx);
    if (key2offset.find(featid) == key2offset.end()) {
      LOG(FATAL) << featid << " not in keys (key2offset)";
    }
    auto offset_count = key2offset[featid];
    CHECK(offset_count.second == 1) 
      << "length of entry != 1 for lr model.";
    auto offset = offset_count.first;
    wTx += weights[offset] * row.get_value(idx);
  }
  if (is_norm) return mit::math::sigmoid(wTx);
  return wTx;
}

void LR::Gradient(const dmlc::Row<mit_uint> & row, 
                  const std::vector<mit_float> & weights,
                  mit::key2offset_type & key2offset,
                  const mit_float & preds, 
                  std::vector<mit_float> * grads) {
  auto max_length = weights.size();
  auto instweight = row.get_weight();
  mit_float residual = preds - row.get_label();
  size_t offset = 0;
  for (auto idx = 0u; idx < row.length; ++idx) {
    auto key = row.get_index(idx);
    CHECK(key2offset.find(key) != key2offset.end()) << 
      "key: " << key << " not in key2offset";
    auto offset_count = key2offset[key];
    offset = offset_count.first;
    CHECK(offset < max_length) << "offset: " 
      << offset << " out of range. max_length: " << max_length;
    CHECK(offset_count.second == 1) << "length of entry != 1 for lr model.";
    (*grads)[offset] += residual * row.get_value(idx) * instweight;
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
