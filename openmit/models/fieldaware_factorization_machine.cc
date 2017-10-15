#include "fieldaware_factorization_machine.h"

namespace mit {

FFM::FFM(const mit::KWArgs & kwargs) {
  this->cli_param_.InitAllowUnknown(kwargs);
}

FFM::~FFM() { // TODO }

void FFM::InitOptimizer(const mit::KWArgs & kwargs) {
  optimizer_.reset(mit::Optimizer::Create(kwargs));
  optimizer_v_.reset(
    mit::Optimizer::Create(kwargs, this->cli_param_.optimizer_v));
}

void FFM::Pull(ps::KVPairs<mit_float> & response, 
               mit::EntryMeta * entry_meta, 
               mit::entry_map_type * weight) {
  for (auto i = 0u; i < response.keys.size(); ++i) {
    ps::Key key = response.keys[i];
    if (weight->find(key) == weight->end()) {
      size_t field_number = 0;
      mit_uint fieldid = 0l;
      if (key > 0l) {  // no bias feature item
        fieldid = mit::DecodeField(key, this->cli_param_.nbit);
        CHECK(fieldid > 0) < "fieldid <= 0 for no bias item is error.";
        field_number = entry_meta->CombineInfo(fieldid)->size();
      }
      mit::Entry * entry = new mit::Entry(
          this->cli_param_, field_number, fieldid);
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
  // TODO
}

void FFM::Gradient(const dmlc::Row<mit_uint> & row, const std::vector<mit_float> & weights, std::unordered_map<mit_uint, std::pair<size_t, int> > & key2offset, const mit_float & preds, std::vector<mit_float> * grads) {
  // TODO 
}

void FFM::Gradient(const dmlc::Row<mit_uint> & row, const mit_float & pred, mit::SArray<mit_float> * grad) {
  // TODO
}

mit_float FFM::Predict(const dmlc::Row<mit_uint> & row, const std::vector<mit_float> & weights, std::unordered_map<mit_uint, std::pair<size_t, int> > & key2offset, bool is_norm) {
  // TODO
  return 0.0f;
}

mit_float FFM::Predict(const dmlc::Row<mit_uint> & row, const mit::SArray<mit_float> & weight, bool is_norm) {
  // TODO 
  return 0.0f;
}

/*
// TODO is_linear? is_sigmoid?
mit_float FFM::Predict(const dmlc::Row<mit_uint> & row, 
                       mit::PMAPT & weight,
                       bool is_norm) {
  mit_float raw_score = RawExpr(row, weight);
  if (is_norm) return mit::math::sigmoid(raw_score);
  return raw_score;
} // method Predict

mit_float FFM::RawExpr(const dmlc::Row<mit_uint> & row, 
                       mit::PMAPT & weight) {
  mit_float wTx = 0.0;
  if (this->param_.is_linear) wTx = Linear(row, weight);
  mit_float cross = Cross(row, weight);
  return wTx + cross;
}

mit_float FFM::Linear(const dmlc::Row<mit_uint> & row,
                      mit::PMAPT & weight) {
  mit_float wTx = weight[0]->Get(0);
  // OpenMP ?
  for (auto i = 0u; i < row.length; ++i) {
    wTx += weight[row.index[i]]->Get(0) * row.value[i];
  }
  return wTx;
}

mit_float FFM::Cross(const dmlc::Row<mit_uint> & row,
                     mit::PMAPT & weight) {
  mit_float cross = 0.0;
  for (auto i = 0u; i < row.length - 1; ++i) {
    auto xi = row.value[i];
    auto feati = row.index[i];
    auto fi = row.field[i];
    for (auto j = i + 1; j < row.length; ++j) {
      auto xj = row.value[j];
      auto featj = row.index[j];
      auto fj = row.field[j];
      auto inner_prod = 0.0f;
      // OpenMP? Eigen?
      for (auto k = 0u; k < param_.k; ++k) {
        auto vifj = weight[feati]->Get(1 + (fj-1)*param_.k + k);
        auto vjfi = weight[featj]->Get(1 + (fi-1)*param_.k + k);
        inner_prod += vifj * vjfi;
      }
      cross += inner_prod * xi * xj;
    }
  }
  return cross;
}

void FFM::Gradient(const dmlc::Row<mit_uint> & row, 
                   const mit_float & pred,
                   mit::PMAPT & weight,
                   mit::PMAPT * grad) {
  auto residual = pred - row.get_label();
  if (this->param_.is_linear) {
    (*grad)[0]->Set(0, residual * 1);   // bias
    // 1-order linear 
    for (auto i = 0u; i < row.length; ++i) {
      mit_uint feati = row.index[i];
      mit_float partial_wi = residual * row.value[i];
      (*grad)[feati]->Set(0, (*grad)[feati]->Get(0) + partial_wi);
    }
  }

  // 2-order cross
  for (auto i = 0u; i < row.length - 1; ++i) {
    auto xi = row.value[i];
    auto feati = row.index[i];
    auto fi = row.field[i];
    for (auto j = i + 1; j < row.length; ++j) {
      auto xj = row.value[j];
      auto featj = row.index[j];
      auto fj = row.field[j];
      
      if (fi == fj) continue;   // same field not cross

      for (auto k = 0u; k < param_.k; ++k) {
        auto index_ifjk = 1 + (fj - 1) * param_.field_num + k;
        auto grad_ifjk = residual * (weight[feati]->Get(index_ifjk) * xi * xj);
        auto index_jfik = 1 + (fi - 1) * param_.field_num + k;
        auto grad_jfik = residual * (weight[featj]->Get(index_jfik) * xi * xj);

        (*grad)[feati]->Set(
            index_ifjk, (*grad)[feati]->Get(index_ifjk) + grad_ifjk);
        (*grad)[featj]->Set(
            index_jfik, (*grad)[featj]->Get(index_jfik) + grad_jfik);
      }
    }
  } 
} // method FFM::Gradient
*/
} // namespace mit
