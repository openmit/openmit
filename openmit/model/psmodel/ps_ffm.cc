#include "openmit/models/ffm.h"

namespace mit {

/*
void FFM::Pull(ps::KVPairs<mit_float> & response, 
               mit::entry_map_type * weight) {
  for (auto i = 0u; i < response.keys.size(); ++i) {
    ps::Key key = response.keys[i];
    if (weight->find(key) == weight->end()) {
      mit_uint fieldid = 0l;
      if (key > 0l) {  // no bias feature item
        fieldid = mit::DecodeField(key, model_param_.nbit);
        CHECK(fieldid > 0) << "fieldid <= 0 for no bias item is error.";
      }
      mit::Entry * entry = mit::Entry::Create(
        model_param_, entry_meta_.get(), random_.get(), fieldid);
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
*/
void FFM::Pull(ps::KVPairs<mit_float>& response, mit::entry_map_type* weight) {
  size_t keys_size = response.keys.size();
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
  std::vector<std::vector<mit_float>* > vals_thread(cli_param_.num_thread);
  for (auto i = 0u; i < cli_param_.num_thread; ++i) {
    vals_thread[i] = new std::vector<mit_float>();
  }
  int blocksize = (keys_size - 1) / cli_param_.num_thread;
  if ((keys_size - 1) % cli_param_.num_thread != 0) blocksize += 1;
  #pragma omp parallel for schedule(static, blocksize)
  for (auto i = 1u; i < keys_size; ++i) {
    ps::Key key = response.keys[i];
    if (weight->find(key) == weight->end()) {
      auto fid = mit::DecodeField(key, model_param_.nbit);  // fid 
      CHECK(fid > 0);
      auto * entry = mit::Entry::Create(
        model_param_, entry_meta_.get(), random_.get(), fid);
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
  for (auto i = 0u; i < cli_param_.num_thread; ++i) {
    ps::SArray<mit_float> sarray(
      vals_thread[i]->data(), vals_thread[i]->size());
    response.vals.append(sarray);
    if (vals_thread[i] != NULL) {
      delete vals_thread[i]; vals_thread[i] = NULL;
    }
  }
}
 
void FFM::Update(const ps::SArray<mit_uint> & keys, 
                 const ps::SArray<mit_float> & vals, 
                 const ps::SArray<int> & lens, 
                 mit::entry_map_type * weight) {
  auto keys_length = keys.size();
  auto offset = 0u;
  for (auto i = 0u; i < keys_length; ++i) {
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
} // Update

void FFM::Gradient(const dmlc::Row<mit_uint> & row, 
                   const std::vector<mit_float> & weights, 
                   mit::key2offset_type & key2offset, 
                   std::vector<mit_float> * grads,
                   const mit_float & lossgrad_value) { 
  auto instweight = row.get_weight();
  // 0-order intercept
  if (! cli_param_.is_contain_intercept) {
    auto offset0 = key2offset[0].first;
    (*grads)[offset0] += lossgrad_value * 1 * instweight;
  }
  // 1-order linear item 
  for (auto i = 0u; i < row.length; ++i) {
    mit_uint key = row.index[i] == 0 ? 0l : 
      mit::NewKey(row.index[i], row.field[i], model_param_.nbit);
    CHECK(key2offset.find(key) != key2offset.end()) << 
      "key: " << key << " not in key2offset";
    auto offset = key2offset[key].first;
    auto xi = row.get_value(i);
    auto partial_wi = lossgrad_value * xi * instweight;
    (*grads)[offset] += partial_wi;
  }
  // 2-order cross item
  for (auto i = 0u; i < row.length - 1; ++i) {
    auto fi = row.field[i];
    auto keyi = row.index[i] == 0 ? 0l : 
      mit::NewKey(row.index[i], fi, model_param_.nbit);
    auto xi = row.get_value(i);
    // fi not in fields_map
    if (entry_meta_->CombineInfo(fi)->size() == 0) continue;
    for (auto j = i + 1; j < row.length; ++j) {
      auto fj = row.field[j];
      if (fi == fj) continue; // not cross when same field 
      if (entry_meta_->CombineInfo(fj)->size() == 0) continue;
      auto keyj = row.index[j] == 0 ? 0l :
        mit::NewKey(row.index[j], fj, model_param_.nbit);
      auto xj = row.get_value(j);

      auto vifj_index = entry_meta_->FieldIndex(fi, fj);
      if (vifj_index == -1) continue;
      auto vifj_offset = key2offset[keyi].first + 
        (1 + vifj_index * model_param_.embedding_size);

      auto vjfi_index = entry_meta_->FieldIndex(fj, fi);
      if (vjfi_index == -1) continue;
      auto vjfi_offset = key2offset[keyj].first + 
        (1 + vjfi_index * model_param_.embedding_size);

      // vifj * vjfi * xi * xj
      for (auto k = 0u; k < model_param_.embedding_size; ++k) {
        (*grads)[vifj_offset+k] += 
          lossgrad_value * (weights[vjfi_offset+k] * xi * xj) * instweight;
        (*grads)[vjfi_offset+k] += 
          lossgrad_value * (weights[vifj_offset+k] * xi * xj) * instweight;
      }
    }
  } 
}

void FFM::Gradient(const dmlc::Row<mit_uint> & row, const mit_float & pred, mit::SArray<mit_float> * grad) {
  // TODO
}

mit_float FFM::Predict(const dmlc::Row<mit_uint> & row, 
                       const std::vector<mit_float> & weights, 
                       mit::key2offset_type & key2offset, 
                       bool is_norm) {
  auto wTx = Linear(row, weights, key2offset);
  wTx += Cross(row, weights, key2offset);
  if (is_norm) return mit::math::sigmoid(wTx);
  return wTx;
}

mit_float FFM::Predict(const dmlc::Row<mit_uint> & row, const mit::SArray<mit_float> & weight, bool is_norm) {
  // TODO 
  return 0.0f;
}

mit_float FFM::Linear(const dmlc::Row<mit_uint> & row, 
                      const std::vector<mit_float> & weights, 
                      mit::key2offset_type & key2offset) {
  mit_float wTx = 0.0f;
  // intercept
  if (! cli_param_.is_contain_intercept) {
    wTx += weights[key2offset[0].first];
  }
  // TODO SMID Accelated
  for (auto i = 0u; i < row.length; ++i) {
    auto keyi = row.index[i];
    auto fid = row.field[i];
    auto newkey = (keyi == 0l) ? 0l :
      mit::NewKey(keyi, fid, model_param_.nbit);
    CHECK(key2offset.find(newkey) != key2offset.end())
      << "newkey: " << newkey << " not in key2offset." 
      << "key: " << keyi << ", field: " << fid;
    auto offseti = key2offset[newkey].first;
    wTx += offseti * row.get_value(i);
  }
  return wTx;
}

mit_float FFM::Cross(const dmlc::Row<mit_uint> & row, 
                     const std::vector<mit_float> & weights, 
                     mit::key2offset_type & key2offset) {
  mit_float cross = 0.0f;
  for (auto i = 0u; i < row.length - 1; ++i) {
    auto fi = row.field[i];
    auto keyi = row.index[i] == 0 ? 0l :
      mit::NewKey(row.index[i], fi, model_param_.nbit);
    auto xi = row.get_value(i);
    // fi not in fields_map
    if (entry_meta_->CombineInfo(fi)->size() == 0) continue;
    for (auto j = i + 1; j < row.length; ++j) {
      auto fj = row.field[j];
      if (fi == fj) continue; // not cross when same field 
      if (entry_meta_->CombineInfo(fj)->size() == 0) continue;
      auto keyj = row.index[j] == 0 ? 0l :
        mit::NewKey(row.index[j], fj, model_param_.nbit);
      auto xj = row.get_value(j);

      auto inprod = 0.0f;

      auto vifj_index = entry_meta_->FieldIndex(fi, fj);
      auto vjfi_index = entry_meta_->FieldIndex(fj, fi);
      if (vifj_index == -1 || vifj_index == -1) continue;

      auto vifj_offset = key2offset[keyi].first + 
        (1 + vifj_index * model_param_.embedding_size);
      auto vjfi_offset = key2offset[keyj].first + 
        (1 + vjfi_index * model_param_.embedding_size);  
      
      // TODO SMID Accelerated
      for (auto k = 0u; k < model_param_.embedding_size; ++k) {
        inprod += weights[vifj_offset+k] * weights[vjfi_offset+k];
      }
      cross += inprod * xi * xj;
    }
  }
  return cross;
}
} // namespace mit
