#include "openmit/models/fieldaware_factorization_machine.h"

namespace mit {

FFM::~FFM() { 
  // TODO 
}

void FFM::InitOptimizer(const mit::KWArgs & kwargs) {
  optimizer_.reset(mit::Optimizer::Create(kwargs));
  optimizer_v_.reset(
    mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
}

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
 
void FFM::Update(const ps::SArray<mit_uint> & keys, const ps::SArray<mit_float> & vals, const ps::SArray<int> & lens, mit::entry_map_type * weight) {
  auto keys_length = keys.size();
  auto offset = 0u;
  for (auto i = 0u; i < keys_length; ++i) {
    if (weight->find(keys[i]) == weight->end()) {
      LOG(FATAL) << keys[i] << " not in weight structure";
    }
    // update_w (1-order linear item)
    auto w = (*weight)[keys[i]]->Get(0);
    auto g = vals[offset++];
    optimizer_->Update(keys[i], 0, g, w, (*weight)[keys[i]]);
    (*weight)[keys[i]]->Set(0, w);
    // update_v (2-order cross item)
    if (lens[i] == 1) continue;
    for (int idx = 1; idx < lens[i]; ++idx) {
      auto w = (*weight)[keys[i]]->Get(idx);
      auto g = vals[offset++]; 
      optimizer_v_->Update(keys[i], idx, g, w, (*weight)[keys[i]]);
      (*weight)[keys[i]]->Set(idx, w);
    }
  }
} // Update

void FFM::Gradient(const dmlc::Row<mit_uint> & row, 
                   const std::vector<mit_float> & weights, 
                   mit::key2offset_type & key2offset, 
                   const mit_float & pred, 
                   std::vector<mit_float> * grads) {
  auto max_length = weights.size();
  auto instweight = row.get_weight();
  auto residual = pred - row.get_label();
  // 1-order linear item 
  for (auto i = 0u; i < row.length; ++i) {
    mit_uint key = row.index[i] == 0 ? 0l : 
      mit::NewKey(row.index[i], row.field[i], model_param_.nbit);
    
    CHECK(key2offset.find(key) != key2offset.end()) << 
      "key: " << key << " not in key2offset";
    auto offset = key2offset[key].first;
    CHECK(offset < max_length) << "offset: " << offset << 
      " out of range. max_length: " << max_length;
    auto partial_wi = residual * row.get_value(i) * instweight;
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
          (weights[vjfi_offset+k] * xi * xj) * residual * instweight;
        (*grads)[vjfi_offset+k] += 
          (weights[vifj_offset+k] * xi * xj) * residual * instweight;
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
  bool is_exist_bias_index_in_row = false;
  // TODO SMID Accelated
  for (auto i = 0u; i < row.length; ++i) {
    auto key = 0l;
    if (row.index[i] != 0l) {
      key = mit::NewKey(
        row.index[i], row.field[i], model_param_.nbit);
    } else {
      is_exist_bias_index_in_row = true;
    }
    auto wi = key2offset.find(key) == key2offset.end() ? 
      0.0 : weights[key2offset[key].first];
    wTx += wi * row.get_value(i);
  }
  if (! is_exist_bias_index_in_row) {
    if (key2offset.find(0) == key2offset.end()) {
      LOG(FATAL) << "bias item (key=0) not in key2offset";
    }
    wTx += weights[key2offset[0].first];
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
      if (vifj_index == -1) continue;
      auto vifj_offset = key2offset[keyi].first + 
        (1 + vifj_index * cli_param_.embedding_size);

      auto vjfi_index = entry_meta_->FieldIndex(fj, fi);
      if (vifj_index == -1) continue;
      auto vjfi_offset = key2offset[keyj].first + 
        (1 + vjfi_index * cli_param_.embedding_size);  
      
      // TODO SMID Accelated
      for (auto k = 0u; k < cli_param_.embedding_size; ++k) {
        inprod += weights[vifj_offset+k] * weights[vjfi_offset+k];
      }
      cross += inprod * xi * xj;
    }
  }
  return cross;
}
} // namespace mit
