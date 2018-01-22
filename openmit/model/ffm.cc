#include "openmit/model/ffm.h"
#include "openmit/tools/dstruct/sarray.h"

namespace mit {

FFM::FFM(const mit::KWArgs& kwargs) : Model(kwargs) {
  optimizer_v_.reset(
    mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
  CHECK(model_param_.embedding_size > 0);
  blocksize = model_param_.embedding_size / 4 * 4;
  remainder = model_param_.embedding_size % 4;
}

FFM::~FFM() {}

FFM* FFM::Get(const mit::KWArgs& kwargs) {
  return new FFM(kwargs);
}

/*
// single thread
void FFM::Pull(ps::KVPairs<mit_float>& response, mit::entry_map_type* weight) {
  size_t keys_size = response.keys.size();
  CHECK(keys_size > 0);
  CHECK_EQ(keys_size, response.extras.size());   // store field id
  response.lens.resize(keys_size);
  // key 
  if (cli_param_.debug) LOG(INFO) << "FFM::Pull begin";
  for (auto i = 0u; i < keys_size; ++i) {
    ps::Key key = response.keys[i];
    mit::Entry* entry = nullptr;
    if (weight->find(key) == weight->end()) {
      if (cli_param_.debug) LOG(INFO) << "FFM::Pull 1 key not in weight-" << key;
      auto fid = response.extras[i];
      if (key > 0) CHECK(fid > 0) << "fid = 0, key: " << key;
      entry = mit::Entry::Create(model_param_, entry_meta_.get(), random_.get(), fid);
      {
        //std::lock_guard<std::mutex> lk(mu_);
        mu_.lock();
        weight->insert(std::make_pair(key, entry));
        mu_.unlock();
      }
    } else {
      if (cli_param_.debug) LOG(INFO) << "FFM::Pull 1 key in weight-" << key;
      entry = (*weight)[key];
    }
    if (cli_param_.debug) LOG(INFO) << "FFM::Pull 2 key: " << key;
    CHECK_NOTNULL(entry); CHECK_GT(entry->Size(), 0);
    for (auto i = 0u; i < entry->Size(); ++i) {
      response.vals.push_back(entry->Get(i));
    }
    response.lens[i] = entry->Size();
    if (cli_param_.debug) LOG(INFO) << "FFM::Pull 3 key: " << key;
  }
  if (cli_param_.debug) LOG(INFO) << "FFM::Pull done";
}
*/

void FFM::Pull(ps::KVPairs<mit_float>& response, mit::entry_map_type* weight) {
  size_t keys_size = response.keys.size();
  CHECK(keys_size > 0);
  CHECK_EQ(keys_size, response.extras.size());   // store field id
  response.lens.resize(keys_size);

  // feature (multi-thread)
  auto nthread = cli_param_.num_thread; CHECK(nthread > 0);
  int chunksize = keys_size / nthread;
  if (keys_size % nthread != 0) chunksize += 1;
  std::vector<std::vector<mit_float>* > vals_thread(nthread);
  for (auto i = 0u; i < nthread; ++i) {
    vals_thread[i] = new std::vector<mit_float>();
    vals_thread[i]->reserve(chunksize * 5);
  }
  #pragma omp parallel for num_threads(nthread) schedule(static, chunksize)
  for (auto i = 0u; i < keys_size; ++i) {
    int threadid = omp_get_thread_num();
    ps::Key key = response.keys[i];
    mit::Entry* entry = nullptr;
    if (weight->find(key) == weight->end()) {
      auto fid = response.extras[i];
      if (key > 0) CHECK(fid > 0);
      entry = mit::Entry::Create(model_param_, entry_meta_.get(), random_.get(), fid);
      CHECK_NOTNULL(entry);
      weight->insert(std::make_pair(key, entry));
      /*
      #pragma omp critical 
      {
        weight->insert(std::make_pair(key, entry));
      }
      */
    } else {
      entry = (*weight)[key];
    }
    CHECK_NOTNULL(entry); CHECK_GT(entry->Size(), 0);
    vals_thread[threadid]->insert(
        vals_thread[threadid]->end(), entry->Data(), entry->Data() + entry->Size());
    response.lens[i] = entry->Size();
  }

  // merge multi-thread result
  for (auto i = 0u; i < nthread; ++i) {
    ps::SArray<mit_float> sarray(vals_thread[i]->data(), vals_thread[i]->size());
    response.vals.append(sarray);
    delete vals_thread[i]; vals_thread[i] = NULL;
  }
}
 
void FFM::Update(const ps::SArray<mit_uint>& keys, 
                   const ps::SArray<mit_float>& vals, 
                   const ps::SArray<int>& lens, 
                   mit::entry_map_type* weight) {
  auto keys_size = keys.size();
  auto offset = 0u;
  for (auto i = 0u; i < keys_size; ++i) {
    auto key = keys[i];
    CHECK(weight->find(key) != weight->end()) << key << " not in weight";
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
  CHECK_EQ(offset, vals.size()) << "no match between offset and vals.size";
} // FFM::Update

void FFM::Gradient(const dmlc::Row<mit_uint>& row, 
                     const std::vector<mit_float>& weights, 
                     mit::key2offset_type& key2offset, 
                     std::vector<mit_float>* grads, 
                     const mit_float& loss_grad) { 
  mit_float instweight = row.get_weight();
  mit_float middle = loss_grad * instweight;
  // 0-order intercept
  if (! cli_param_.is_contain_intercept) {
    auto offset0 = key2offset[0].first;
    (*grads)[offset0] += 1 * middle;
  }
  // 1-order linear item 
  //#pragma omp parallel for num_threads(cli_param_.num_thread)
  for (auto i = 0u; i < row.length; ++i) {
    mit_uint key = row.index[i];
    CHECK(key2offset.find(key) != key2offset.end());
    auto offset = key2offset[key].first;
    auto xi = row.get_value(i);
    (*grads)[offset] += xi * middle;
  }
  // 2-order cross item 
  //#pragma omp parallel for num_threads(cli_param_.num_thread)
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
      
      auto xij_middle = xi * xj * middle;
      // sse implementation
      //(*grads)[vifj_offset+k] += loss_grad * (weights[vjfi_offset+k] * xi * xj) * instweight;
      //GradientEmbeddingWithSSE(weights.data() + vjfi_offset, grads->data() + vifj_offset, xij_middle);
      //GradientEmbeddingWithSSE(weights.data() + vifj_offset, grads->data() + vjfi_offset, xij_middle);

      for (auto k = 0u; k < model_param_.embedding_size; ++k) {
        (*grads)[vifj_offset + k] += weights[vjfi_offset + k] * xij_middle;
        (*grads)[vjfi_offset + k] += weights[vifj_offset + k] * xij_middle;
      }
    }
  } 
}

mit_float FFM::Predict(const dmlc::Row<mit_uint>& row, 
                       const std::vector<mit_float>& weights, 
                       mit::key2offset_type& key2offset) {
  auto wTx = Linear(row, weights, key2offset);
  wTx += Cross(row, weights, key2offset);
  return wTx;
}

mit_float FFM::Linear(const dmlc::Row<mit_uint>& row, 
                        const std::vector<mit_float>& weights, 
                        mit::key2offset_type& key2offset) {
  mit_float wTx = 0.0f;
  // intercept 
  auto keyintercept = 0l;
  if (! cli_param_.is_contain_intercept) {
    wTx += weights[key2offset[keyintercept].first];
  }
  //#pragma omp parallel for reduction(+:wTx) num_threads(cli_param_.num_thread)
  for (auto i = 0u; i < row.length; ++i) {
    auto key = row.index[i];
    if (! cli_param_.is_contain_intercept && key == 0) continue;
    CHECK(key2offset.find(key) != key2offset.end());
    auto offseti = key2offset[key].first;
    wTx += weights[offseti] * row.get_value(i);
  }
  return wTx;
}

mit_float FFM::Cross(const dmlc::Row<mit_uint>& row, const std::vector<mit_float>& weights, mit::key2offset_type& key2offset) {
  const mit_float* pweights = weights.data();
  size_t weights_size = weights.size();
  mit_float cross = 0.0f;

  //#pragma omp parallel for reduction(+:cross) num_threads(cli_param_.num_thread)
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

      CHECK_LE(vifj_offset + model_param_.embedding_size, weights_size);
      CHECK_LE(vjfi_offset + model_param_.embedding_size, weights_size);

      // sse acceleration
      auto inprod = InnerProductWithSSE(pweights + vifj_offset, pweights + vjfi_offset);
      cross += inprod * xi * xj;

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

} // namespace mit
