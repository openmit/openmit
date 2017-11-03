#include "openmit/models/fm.h"
#include <cmath>

namespace mit {

void FM::InitOptimizer(const mit::KWArgs & kwargs) {
  optimizer_.reset(mit::Optimizer::Create(kwargs));
  optimizer_v_.reset(
    mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
}

void FM::Update(const ps::SArray<mit_uint> & keys, 
                const ps::SArray<mit_float> & vals, 
                const ps::SArray<int> & lens, 
                mit::entry_map_type * weight) {
  auto len_lens_item = 1 + model_param_.embedding_size;
  auto keys_length = keys.size();
  auto offset = 0u;
  for (auto i = 0u; i < keys_length; ++i) {
    auto key = keys[i];
    CHECK(weight->find(key) != weight->end())
      << key << " not in model structure";
    // update_w (1-order linear item)
    auto w = (*weight)[keys[i]]->Get();
    auto g = vals[offset++];
    optimizer_->Update(key, 0, g, w, (*weight)[key]);
    (*weight)[key]->Set(0, w);
    if (keys[i] == 0) continue;
    CHECK_EQ(lens[i], len_lens_item) 
      << "lens[i] != 1+embedding_size for fm model";
    // update_v (2-order cross item)
    for (int k = 1; k < lens[i]; ++k) {
      auto v = (*weight)[key]->Get(k);
      auto g = vals[offset++];
      optimizer_v_->Update(key, k, g, v, (*weight)[key]);
      (*weight)[key]->Set(k, v);
    }
  }
}

void FM::Pull(ps::KVPairs<mit_float> & response, 
              mit::entry_map_type * weight) {
  for (auto i = 0u; i < response.keys.size(); ++i) {
    ps::Key key = response.keys[i];
    ps::SArray<mit_float> mvalue;;
    if (weight->find(key) == weight->end()) {
      mit::Entry * entry = mit::Entry::Create(
        model_param_, entry_meta_.get(), random_.get());
      weight->insert(std::make_pair(key, entry));
    }
    mit::Entry * entry = (*weight)[key];
    if (key == 0) {  // intercept
      mvalue.push_back(entry->Get());
    } else {
      mvalue.CopyFrom(entry->Data(), entry->Size());
    }
    // fill vals and lens of response
    response.vals.append(mvalue);
    response.lens.push_back(mvalue.size());
  }
}

mit_float FM::Predict(const dmlc::Row<mit_uint> & row, 
                      const std::vector<mit_float> & weights, 
                      mit::key2offset_type & key2offset, 
                      bool is_norm) {
  auto wTx = Linear(row, weights, key2offset);
  wTx += Cross(row, weights, key2offset);
  if (is_norm) return mit::math::sigmoid(wTx);
  return wTx;
}

mit_float FM::Linear(const dmlc::Row<mit_uint> & row, 
                     const std::vector<mit_float> & weights, 
                     mit::key2offset_type & key2offset) {
  mit_float wTx = 0.0f;
  if (! cli_param_.is_contain_intercept) {
    auto offset0 = key2offset[0].first;
    wTx += weights[offset0];
  }
  // TODO SMID Accelerated
  for (auto i = 0u; i < row.length; ++i) {
    auto key = row.index[i];
    if (key == 0) continue;
    CHECK(key2offset.find(key) != key2offset.end())
      << "key: " << key << " not in key2offset";
    auto wi = weights[key2offset[key].first];
    wTx += wi * row.get_value(i);
  }
  return wTx;
}

mit_float FM::Cross(const dmlc::Row<mit_uint> & row, 
                    const std::vector<mit_float> & weights, 
                    mit::key2offset_type & key2offset) {
  mit_float cross = 0.0f;
  auto embedsize = model_param_.embedding_size;
  for (auto k = 0u; k < embedsize; ++k) {
    mit_float linsum_quad = 0.0;
    mit_float quad_linsum = 0.0;
    for (auto i = 0u; i < row.length; ++i) {
      auto key = row.index[i];
      auto xi = row.get_value(i);
      if (key == 0) continue;
      auto offset_count = key2offset[key];
      CHECK_EQ(offset_count.second, 1 + embedsize) 
        << "lens[i] != 1 + embedding_size for fm model ." 
        << "is error. and key: " << key;
      auto offset = offset_count.first;
      auto vik = weights[offset + 1 + k];
      linsum_quad += vik * xi;
      quad_linsum += vik * vik * xi * xi;
    }
    cross += (linsum_quad * linsum_quad - quad_linsum);
  }
  return cross;
}

mit_float FM::Predict(const dmlc::Row<mit_uint> & row,
                      const mit::SArray<mit_float> & weight,
                      bool is_norm) {
  // TODO
  return 0.0f;
}

/**
 * for w0: lossgrad_value * 1
 * for wi: lossgrad_value * xi
 * for w(i,f): lossgrad_value * (xi * \sum_{j=1}^{n} (v(j,f) * xj) - v(i,f) * xi^2)
 */
void FM::Gradient(const dmlc::Row<mit_uint> & row, 
                  const std::vector<mit_float> & weights, 
                  mit::key2offset_type & key2offset, 
                  std::vector<mit_float> * grads, 
                  const mit_float & lossgrad_value) {
  CHECK_EQ(weights.size(), grads->size());
  auto instweight = row.get_weight();
  // 0-order intercept 
  if (! cli_param_.is_contain_intercept) {
    auto offset0 = key2offset[0].first;
    (*grads)[offset0] += lossgrad_value * 1 * instweight; 
  }
  // TODO SMID Accelerated 
  // 1-order linear item
  auto embedsize = model_param_.embedding_size;
  for (auto i = 0u; i < row.length; ++i) {
    auto xi = row.get_value(i);
    auto key = row.index[i];
    auto offset = key2offset[key].first;
    auto partial_wi = lossgrad_value * xi * instweight;
    (*grads)[offset] += partial_wi;
  }
  // 2-order cross item 
  std::vector<mit_float> sum(embedsize, 0.0f);
  for (auto k = 0u; k < embedsize; ++k) {
    for (auto j = 0u; j < row.length; ++j) {
      auto xj = row.get_value(j);
      auto offsetj = key2offset[row.index[j]].first;
      sum[k] += weights[offsetj + 1 + k] * xj;
    }
  }
  for (auto i = 0u; i < row.length; ++i) {
    auto keyi = row.index[i];
    auto xi = row.get_value(i);
    auto offseti = key2offset[keyi].first;
    for (auto k = 0u; k < embedsize; ++k) {
      auto gik = sum[k] - weights[offseti + 1 + k] * xi;
      auto partial_wik = lossgrad_value * xi * gik * instweight;
      (*grads)[offseti + 1 + k] += partial_wik;
    }
  }
} // FM::Gradient

void FM::Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::SArray<mit_float> * grad) {
  // TODO
}

} // namespace mit 
