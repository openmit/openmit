#include "openmit/model/fm.h"

namespace mit {

/////////////////////////////////////////////////////////////
// fm model complemention for mpi or local
/////////////////////////////////////////////////////////////

FM::~FM() {}

FM* FM::Get(const mit::KWArgs& kwargs) {
  return new FM(kwargs);
}

void FM::Gradient(const dmlc::Row<mit_uint>& row, const mit_float& pred, mit::SArray<mit_float>* grad) {
  // TODO
} // FM::Gradient

mit_float FM::Predict(const dmlc::Row<mit_uint>& row, const mit::SArray<mit_float>& weight, bool norm) {
  // TODO
  return 0.0;
} // FM::Predict

/////////////////////////////////////////////////////////////
// fm model complemention for parameter server framework
/////////////////////////////////////////////////////////////

PSFM::~PSFM() {}

PSFM* PSFM::Get(const mit::KWArgs& kwargs) {
  return new PSFM(kwargs);
}

void PSFM::Update(const ps::SArray<mit_uint> & keys, 
                const ps::SArray<mit_float> & vals, 
                const ps::SArray<int> & lens, 
                mit::entry_map_type * weight) {
  auto entry_size = 1 + model_param_.embedding_size;
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
    CHECK_EQ(lens[i], entry_size) 
      << "lens[i] != 1+embedding_size for fm model";
    // update_v (2-order cross item)
    for (int k = 1; k < lens[i]; ++k) {
      auto v = (*weight)[key]->Get(k);
      auto g = vals[offset++];
      optimizer_v_->Update(key, k, g, v, (*weight)[key]);
      (*weight)[key]->Set(k, v);
    }
  }
  CHECK_EQ(offset, vals.size()) << "offset not match vals.size";
}

void PSFM::Pull(ps::KVPairs<mit_float> & response, 
              mit::entry_map_type * weight) {
  size_t entry_size = 1 + model_param_.embedding_size;
  size_t keys_size = response.keys.size();
  size_t vals_size = 1 + (keys_size - 1) * entry_size;
  response.vals.resize(vals_size);
  response.lens.resize(keys_size, entry_size);
  
  // intercept 
  CHECK(response.keys[0] == 0) << "index of intercept != 0.";
  if (weight->find(0) == weight->end()) {
    auto * entry = mit::Entry::Create(
      model_param_, entry_meta_.get(), random_.get());
    weight->insert(std::make_pair(0, entry));
  }
  response.vals[0] = (*weight)[0]->Get();
  response.lens[0] = 1;

  // feature
  omp_set_num_threads(cli_param_.num_thread);
  size_t blocksize = (keys_size - 1) / cli_param_.num_thread;
  if ((keys_size - 1) % cli_param_.num_thread != 0) blocksize += 1;
  #pragma omp parallel for schedule(static, blocksize)
  for (auto i = 1u; i < keys_size; ++i) {
    ps::Key key = response.keys[i];
    if (weight->find(key) == weight->end()) {
      mit::Entry * entry = mit::Entry::Create(
        model_param_, entry_meta_.get(), random_.get());
      #pragma omp critical
      weight->insert(std::make_pair(key, entry));
    }
    mit::Entry * entry = (*weight)[key];
    for (auto idx = 0u; idx < entry_size; ++idx) {
      auto index = 1 + (i - 1) * entry_size + idx;
      response.vals[index] = entry->Get(idx);
    }
  }
}

mit_float PSFM::Predict(const dmlc::Row<mit_uint>& row, 
                        const std::vector<mit_float>& weights, 
                        mit::key2offset_type& key2offset, 
                        bool norm) {
  auto wTx = Linear(row, weights, key2offset);
  wTx += Cross(row, weights, key2offset);
  if (norm) return mit::math::sigmoid(wTx);
  return wTx;
} // PSFM::Predict

mit_float PSFM::Linear(const dmlc::Row<mit_uint>& row, 
                       const std::vector<mit_float>& weights, 
                       mit::key2offset_type& key2offset) {
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

mit_float PSFM::Cross(const dmlc::Row<mit_uint>& row, 
                      const std::vector<mit_float>& weights, 
                      mit::key2offset_type& key2offset) {
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

/**
 * for w0: lossgrad_value * 1
 * for wi: lossgrad_value * xi
 * for w(i,f): lossgrad_value * (xi * \sum_{j=1}^{n} (v(j,f) * xj) - v(i,f) * xi^2)
 */
void PSFM::Gradient(const dmlc::Row<mit_uint> & row, 
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
} // PSFM::Gradient

} // namespace mit 
