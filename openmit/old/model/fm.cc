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

mit_float FM::Predict(const dmlc::Row<mit_uint>& row, const mit::SArray<mit_float>& weight) {
  // TODO
  return 0.0;
} // FM::Predict

/////////////////////////////////////////////////////////////
// fm model complemention for parameter server framework
/////////////////////////////////////////////////////////////

PSFM::PSFM(const mit::KWArgs& kwargs) : PSModel(kwargs) {
  optimizer_v_.reset(
    mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
}

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

void PSFM::Pull(ps::KVPairs<mit_float>& response, mit::entry_map_type* weight) {
  size_t entry_size = 1 + model_param_.embedding_size;
  size_t keys_size = response.keys.size();
  response.lens.resize(keys_size, entry_size);
  size_t key0_size = response.keys[0] == 0l ? 1 : entry_size;
  size_t vals_size = key0_size + (keys_size - 1) * entry_size;
  response.vals.resize(vals_size);
  if (key0_size == 1) response.lens[0] = 1;
  
  // keys (multi-thread)
  auto nthread = cli_param_.num_thread; CHECK(nthread > 0);
  size_t chunksize = keys_size / cli_param_.num_thread;
  if (keys_size % cli_param_.num_thread != 0) chunksize += 1;
  #pragma omp parallel for num_threads(nthread) schedule(static, chunksize)
  for (auto i = 0u; i < keys_size; ++i) {
    ps::Key key = response.keys[i];
    mit::Entry* entry = nullptr;
    if (weight->find(key) == weight->end()) {
      entry = mit::Entry::Create(model_param_, entry_meta_.get(), random_.get());
      #pragma omp critical
      {
        std::lock_guard<std::mutex> lk(mu_);
        weight->insert(std::make_pair(key, entry));
      }
    } else {
      entry = (*weight)[key];
    }
    CHECK_NOTNULL(entry); CHECK_GE(entry->Size(), 1);
    int offset = key0_size + (i-1) * entry_size;
    if (i == 0) offset = 0;
    for (auto idx = 0u; idx < entry->Size(); ++idx) {
      response.vals[offset + idx] = entry->Get(idx);
    }
  }
}

mit_float PSFM::Predict(const dmlc::Row<mit_uint>& row, 
                        const std::vector<mit_float>& weights, 
                        mit::key2offset_type& key2offset) {
  auto wTx = Linear(row, weights, key2offset);
  wTx += Cross(row, weights, key2offset);
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
    CHECK(key2offset.find(key) != key2offset.end());
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
      CHECK_EQ(offset_count.second, 1 + embedsize);
      auto offset = offset_count.first;
      auto vik = weights[offset + 1 + k];
      auto mul_vik_xi = vik * xi;
      linsum_quad += mul_vik_xi;
      quad_linsum += mul_vik_xi * mul_vik_xi;
    }
    cross += (linsum_quad * linsum_quad - quad_linsum);
  }
  return cross;
}

/**
 * for w0: loss_grad * 1
 * for wi: loss_grad * xi
 * for w(i,f): loss_grad * (xi * \sum_{j=1}^{n} (v(j,f) * xj) - v(i,f) * xi^2)
 */
void PSFM::Gradient(const dmlc::Row<mit_uint>& row, 
                    const std::vector<mit_float>& weights, 
                    mit::key2offset_type& key2offset, 
                    std::vector<mit_float>* grads, 
                    const mit_float& loss_grad) {
  CHECK_EQ(weights.size(), grads->size());
  auto instweight = row.get_weight();
  auto middle = loss_grad * instweight;
  // 0-order intercept 
  if (! cli_param_.is_contain_intercept) {
    auto offset0 = key2offset[0].first;
    (*grads)[offset0] += 1 * middle; 
  }
  // TODO SMID Accelerated 
  // 1-order linear item
  auto embedsize = model_param_.embedding_size;
  for (auto i = 0u; i < row.length; ++i) {
    auto xi = row.get_value(i);
    auto key = row.index[i];
    auto offset = key2offset[key].first;
    auto partial_wi = xi * middle;
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
      auto partial_wik = xi * gik * middle;
      (*grads)[offseti + 1 + k] += partial_wik;
    }
  }
} // PSFM::Gradient

} // namespace mit 
