#include "openmit/models/fm.h"

namespace mit {

void FM::InitOptimizer(const mit::KWArgs & kwargs) {
  optimizer_.reset(mit::Optimizer::Create(kwargs));
  optimizer_v_.reset(
    mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
}

void FM::Pull(ps::KVPairs<mit_float> & response, mit::entry_map_type * weight) {
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
  bool is_exist_bias_index_in_row = false;
  // TODO SMID Accelerated
  for (auto i = 0u; i < row.length; ++i) {
    if (row.index[i] == 0l) {
      is_exist_bias_index_in_row = true;
    }
    auto wi = key2offset.find(row.index[i]) == key2offset.end() ?
      0.0 : weights[key2offset[row.index[i]].first];
    wTx += wi * row.get_value(i);
  }
  if (! is_exist_bias_index_in_row) {
    if (key2offset.find(0) == key2offset.end()) {
      LOG(FATAL) << "intercept (key=0) not in key2offset";
    }
    wTx += weights[key2offset[0].first];
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
      if (key == 0 || 
          key2offset.find(key) == key2offset.end()) {
        continue;
      }
      auto offset_count = key2offset[key];
      CHECK_EQ(offset_count.second, 1 + embedsize) 
        << "lens[i] != 1+embedding_size for fm model";
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
 * for logic loss:  residual = pred - label;
 * for w0: residual * 1
 * for wi: residual * xi
 * for w(i,f): residual * (xi * \sum_{j=1}^{n} (v(j,f) * xj) - v(i,f) * xi^2)
 */
void FM::Gradient(const dmlc::Row<mit_uint> & row, 
                  const std::vector<mit_float> & weights, 
                  mit::key2offset_type & key2offset, 
                  const mit_float & pred, 
                  std::vector<mit_float> * grads) {
  CHECK_EQ(weights.size(), grads->size());
  auto residual = pred - row.get_label();
  bool is_exist_bias_index_in_row = false;
  // TODO SMID Accelerated 
  // linear 
  auto embedsize = model_param_.embedding_size;
  for (auto i = 0u; i < row.length; ++i) {
    auto key = row.index[i];
    if (key == 0l) { 
      is_exist_bias_index_in_row = true;
    }
    if (key2offset.find(key) == key2offset.end()) {
      LOG(FATAL) << key << " not in key2offset";
    }
    auto offset_count = key2offset[key];
    CHECK_EQ(offset_count.second, 1 + embedsize) 
      << "lens[i] != 1+embedding_size for fm model";
    auto offset = offset_count.first;
    CHECK(offset < weights.size()) 
      << "offset out of boundary. " << offset;
    auto partial_wi = residual * row.get_value(i);
    (*grads)[offset] += partial_wi;
  }
  // 0-order intercept
  if (! is_exist_bias_index_in_row) {
    (*grads)[key2offset[0].first] += residual * 1;
  }
  // 2-order cross item 
  for (auto i = 0u; i < row.length; ++i) {
    auto keyi = row.index[i];
    auto xi = row.get_value(i);
    for (auto k = 0u; k < embedsize; ++k) {
      auto sum = 0.0f;
      for (auto j = 0u; j < row.length; ++j) {
        if (i == j) continue;
        auto keyj = row.index[j];
        auto xj = row.get_value(j);

        auto offsetj = key2offset[keyj].first;
        sum += weights[offsetj + 1 + k] * xj;
      }
      auto partial_wik = residual * (xi * sum);
      auto offseti = key2offset[keyi].first;
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
