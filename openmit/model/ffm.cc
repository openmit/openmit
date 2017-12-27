#include <pmmintrin.h>
#include "openmit/model/ffm.h"

namespace mit {

/////////////////////////////////////////////////////////////
// ffm model complemention for parameter server framework
/////////////////////////////////////////////////////////////

PSFFM::~PSFFM() {}

PSFFM* PSFFM::Get(const mit::KWArgs& kwargs) {
  return new PSFFM(kwargs);
}
/*
// single thread
void PSFFM::Pull(ps::KVPairs<mit_float>& response, mit::entry_map_type* weight) {
  size_t keys_size = response.keys.size();
  CHECK(keys_size > 0);
  CHECK_EQ(keys_size, response.extras.size());   // store field id
  response.lens.resize(keys_size);

  // intercept
  CHECK_EQ(response.keys[0], 0);
  if (weight->find(0) == weight->end()) {
    weight->insert(std::make_pair(0, mit::Entry::Create(
      model_param_, entry_meta_.get(), random_.get(), 0)));
  }
  response.vals.push_back((*weight)[0]->Get());
  response.lens[0] = 1;
  
  // feature 
  for (auto i = 1u; i < keys_size; ++i) {
    ps::Key key = response.keys[i];
    if (weight->find(key) == weight->end()) {
      auto fid = response.extras[i];
      CHECK(fid > 0) << "fid = 0";
      auto* entry = mit::Entry::Create(model_param_, entry_meta_.get(), random_.get(), fid);
      weight->insert(std::make_pair(key, entry));
    }
    auto* entry = (*weight)[key];
    ps::SArray<mit_float> entry_data(entry->Data(), entry->Size());
    response.vals.append(entry_data);
    response.lens[i] = entry->Size();
  }
}
*/

void PSFFM::Pull(ps::KVPairs<mit_float>& response, mit::entry_map_type* weight) {
  size_t keys_size = response.keys.size();
  CHECK(keys_size > 0);
  CHECK_EQ(keys_size, response.extras.size());   // store field id
  response.lens.resize(keys_size);

  // intercept
  CHECK_EQ(response.keys[0], 0);
  if (weight->find(0) == weight->end()) {
    weight->insert(std::make_pair(0, mit::Entry::Create(
      model_param_, entry_meta_.get(), random_.get(), 0)));
  }
  response.vals.push_back((*weight)[0]->Get());
  response.lens[0] = 1;

  // feature (multi-thread)
  auto nthread = cli_param_.num_thread; CHECK(nthread > 0);
  std::vector<std::vector<mit_float>* > vals_thread(nthread);
  for (auto i = 0u; i < nthread; ++i) {
    vals_thread[i] = new std::vector<mit_float>();
  }
  int chunksize = (keys_size - 1) / nthread;
  if ((keys_size - 1) % nthread != 0) chunksize += 1;
  #pragma omp parallel for num_threads(nthread) schedule(static, chunksize)
  for (auto i = 1u; i < keys_size; ++i) {
    ps::Key key = response.keys[i];
    if (weight->find(key) == weight->end()) {
      auto fid = response.extras[i]; CHECK(fid > 0);
      auto* entry = mit::Entry::Create(model_param_, entry_meta_.get(), random_.get(), fid);
      #pragma omp critical
      weight->insert(std::make_pair(key, entry));
    }
    auto* entry = (*weight)[key];
    for (auto idx = 0u; idx < entry->Size(); ++idx) {
      vals_thread[omp_get_thread_num()]->push_back(entry->Get(idx));
    }
    response.lens[i] = entry->Size();
  }

  // merge multi-thread result
  for (auto i = 0u; i < nthread; ++i) {
    ps::SArray<mit_float> sarray(vals_thread[i]->data(), vals_thread[i]->size());
    response.vals.append(sarray);
    if (vals_thread[i]) { // free memory 
      vals_thread[i]->clear();
      delete vals_thread[i]; vals_thread[i] = NULL;
    }
  }
}
 
void PSFFM::Update(const ps::SArray<mit_uint>& keys, 
                 const ps::SArray<mit_float>& vals, 
                 const ps::SArray<int>& lens, 
                 mit::entry_map_type* weight) {
  auto keys_size = keys.size();
  auto offset = 0u;
  for (auto i = 0u; i < keys_size; ++i) {
    auto key = keys[i];
    if (weight->find(key) == weight->end()) {
      LOG(FATAL) << key << " not in model structure";
    }
    // update_w (1-order linear item)
    auto w = (*weight)[key]->Get(0);
    auto g = vals[offset++];
    optimizer_->Update(key, 0, g, w, (*weight)[key]);
    (*weight)[key]->Set(0, w);
    // update_v (2-order cross item)
    if (lens[i] == 1) continue;
    for (int k = 1; k < lens[i]; ++k) {
      auto v = (*weight)[key]->Get(k);
      auto g = vals[offset++]; 
      optimizer_v_->Update(key, k, g, v, (*weight)[key]);
      (*weight)[key]->Set(k, v);
    }
  }
  CHECK_EQ(offset, vals.size()) << "offset not match vals.size for model update";
} // PSFFM::Update

void PSFFM::Gradient(const dmlc::Row<mit_uint>& row, 
                     const std::vector<mit_float>& weights, 
                     mit::key2offset_type& key2offset, 
                     std::vector<mit_float>* grads, 
                     const mit_float& loss_grad) { 
  auto instweight = row.get_weight();
  // 0-order intercept
  if (! cli_param_.is_contain_intercept) {
    auto offset0 = key2offset[0].first;
    (*grads)[offset0] += loss_grad * 1 * instweight;
  }
  // 1-order linear item 
  #pragma omp parallel for num_threads(cli_param_.num_thread)
  for (auto i = 0u; i < row.length; ++i) {
    mit_uint key = row.index[i];
    CHECK(key2offset.find(key) != key2offset.end()) << key << " not in key2offset";
    auto offset = key2offset[key].first;
    auto xi = row.get_value(i);
    (*grads)[offset] += loss_grad * xi * instweight;
  }
  // 2-order cross item 
  #pragma omp parallel for num_threads(cli_param_.num_thread)
  for (auto i = 0u; i < row.length - 1; ++i) {
    auto fi = row.field[i];
    auto keyi = row.index[i];
    auto xi = row.get_value(i);
    // fi not in fields_map 
    if (! entry_meta_->CombineInfo(fi)) continue;

    for (auto j = i + 1; j < row.length; ++j) {
      auto fj = row.field[j];
      if (fi == fj) continue; // not cross when same field 
      if (! entry_meta_->CombineInfo(fj)) continue;
      auto keyj = row.index[j];
      auto xj = row.get_value(j);

      auto vifj_index = entry_meta_->FieldIndex(fi, fj);
      if (vifj_index == -1) continue;
      auto vifj_offset = key2offset[keyi].first + (1 + vifj_index * model_param_.embedding_size);

      auto vjfi_index = entry_meta_->FieldIndex(fj, fi);
      if (vjfi_index == -1) continue;
      auto vjfi_offset = key2offset[keyj].first + (1 + vjfi_index * model_param_.embedding_size);

      // vifj * vjfi * xi * xj
      // SSE Accelerated ?  const initialize for sse
      for (auto k = 0u; k < model_param_.embedding_size; ++k) {
        (*grads)[vifj_offset+k] += loss_grad * (weights[vjfi_offset+k] * xi * xj) * instweight;
        (*grads)[vjfi_offset+k] += loss_grad * (weights[vifj_offset+k] * xi * xj) * instweight;
      }
    }
  } 
}

mit_float PSFFM::Predict(const dmlc::Row<mit_uint>& row, 
                       const std::vector<mit_float>& weights, 
                       mit::key2offset_type& key2offset, 
                       bool norm) {
  auto wTx = Linear(row, weights, key2offset);
  wTx += Cross(row, weights, key2offset);
  if (norm) return mit::math::sigmoid(wTx);
  return wTx;
}

mit_float PSFFM::Linear(const dmlc::Row<mit_uint>& row, 
                        const std::vector<mit_float>& weights, 
                        mit::key2offset_type& key2offset) {
  mit_float wTx = 0.0f;
  // intercept
  if (! cli_param_.is_contain_intercept) {
    wTx += weights[key2offset[0].first];
  }
  #pragma omp parallel for reduction(+: wTx) num_threads(cli_param_.num_thread)
  for (auto i = 0u; i < row.length; ++i) {
    auto key = row.index[i];
    CHECK(key2offset.find(key) != key2offset.end()) << key << " not in key2offset";
    auto offseti = key2offset[key].first;
    wTx += weights[offseti] * row.get_value(i);
  }
  return wTx;
}

mit_float PSFFM::Cross(const dmlc::Row<mit_uint>& row, const std::vector<mit_float>& weights, mit::key2offset_type& key2offset) {
  const mit_float* pweights = weights.data();
  // for sse instructor
  auto cntBlock = model_param_.embedding_size / 4;
  auto remainder = model_param_.embedding_size % 4;
  __m128 inprod = _mm_setzero_ps();

  mit_float cross = 0.0f;

  #pragma omp parallel for reduction(+: cross) num_threads(cli_param_.num_thread)
  for (auto i = 0u; i < row.length - 1; ++i) {
    auto fi = row.field[i];
    auto keyi = row.index[i];
    auto xi = row.get_value(i);
    // fi not in fields_map 

    if (! entry_meta_->CombineInfo(fi)) continue;
    for (auto j = i + 1; j < row.length; ++j) {
      auto fj = row.field[j];
      if (fi == fj) continue; // not cross when same field 
      if (! entry_meta_->CombineInfo(fj)) continue;
      auto keyj = row.index[j];
      auto xj = row.get_value(j);


      auto vifj_index = entry_meta_->FieldIndex(fi, fj);
      auto vjfi_index = entry_meta_->FieldIndex(fj, fi);
      if (vifj_index == -1 || vifj_index == -1) continue;

      auto vifj_offset = key2offset[keyi].first + (1 + vifj_index * model_param_.embedding_size);
      auto vjfi_offset = key2offset[keyj].first + (1 + vjfi_index * model_param_.embedding_size);
      
      // SMID Accelerated 
      std::vector<mit_float> vifj(pweights + vifj_offset, pweights + vifj_offset + model_param_.embedding_size);
      std::vector<mit_float> vjfi(pweights + vjfi_offset, pweights + vjfi_offset + model_param_.embedding_size);
      
      inprod = _mm_setzero_ps();
      auto offset = 0u;
      for (; offset < cntBlock * 4; offset += 4) {
        __m128 v1 = _mm_load_ps(vifj.data() + offset);
        __m128 v2 = _mm_load_ps(vjfi.data() + offset);
        inprod = _mm_add_ps(inprod, _mm_mul_ps(v1, v2));
      }

      inprod = _mm_hadd_ps(inprod, inprod);
      inprod = _mm_hadd_ps(inprod, inprod);
      mit_float v;
      _mm_store_ss(&v, inprod);

      if (remainder != 0) {
        for (auto i = 0u; i < remainder; ++i) {
          v += vifj[offset + i] * vjfi[offset + i];
        }
      }
      cross += v * xi * xj;

      /*
      auto inprod = 0.0f;
      for (auto k = 0u; k < model_param_.embedding_size; ++k) {
        inprod += weights[vifj_offset+k] * weights[vjfi_offset+k];
      }
      cross += inprod * xi * xj;
      */
    }
  }
  return cross;
}

/////////////////////////////////////////////////////////////
// ffm model complemention for mpi or local
/////////////////////////////////////////////////////////////

FFM::~FFM() {}

FFM* FFM::Get(const mit::KWArgs& kwargs) {
  return new FFM(kwargs);
} 

void FFM::Gradient(const dmlc::Row<mit_uint>& row, const mit_float& pred, mit::SArray<mit_float>* grad) {
  // TODO
} // FFM::Gradient

mit_float FFM::Predict(const dmlc::Row<mit_uint>& row, const mit::SArray<mit_float>& weight, bool norm) {
  // TODO
  return 0.0f;
} // FFM::Predict

} // namespace mit
