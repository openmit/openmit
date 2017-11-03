#include "openmit/models/fm.h"
#include "openmit/models/ffm.h"
#include "openmit/models/lr.h"
#include "openmit/models/model.h"

namespace mit {
Model * Model::Create(const mit::KWArgs & kwargs) {
  std::string model = "lr";
  for (auto & kv : kwargs) {
    if (kv.first != "model") continue;
    model = kv.second;
  }
  if (model == "lr") {
    return mit::LR::Get(kwargs);
  } else if (model == "fm") {
    return mit::FM::Get(kwargs);
  } else if (model == "ffm") {
    return mit::FFM::Get(kwargs);
  } else {
    LOG(FATAL) <<
      "model not in [lr, fm, ffm], model: " << model;
    return nullptr;
  }
}

Model::Model(const mit::KWArgs & kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  model_param_.InitAllowUnknown(kwargs);
  entry_meta_.reset(new mit::EntryMeta(model_param_));
  random_.reset(mit::math::ProbDistr::Create(model_param_));
}

void Model::Predict(const dmlc::RowBlock<mit_uint> & batch,
                    const std::vector<mit_float> & weights, 
                    key2offset_type & key2offset,
                    std::vector<mit_float> & preds, 
                    bool is_norm) {
  CHECK_EQ(batch.size, preds.size());
  // TODO OpenMP?
  for (auto i = 0u; i < batch.size; ++i) {
    preds[i] = Predict(batch[i], weights, key2offset, is_norm);
    //LOG(INFO) << "pred: " << preds[i] << ", label: " << batch[i].get_label();
  }
}

void Model::Predict(const dmlc::RowBlock<mit_uint> & batch, 
                    mit::SArray<mit_float> & weight,
                    std::vector<mit_float> * preds, 
                    bool is_norm) {
  CHECK_EQ(batch.size, preds->size());
  // TODO OpenMP?
  for (auto i = 0u; i < batch.size; ++i) {
    (*preds)[i] = Predict(batch[i], weight, is_norm);
  }
} // method Predict

void Model::Gradient(const dmlc::RowBlock<mit_uint> & batch, std::vector<mit_float> & preds, mit::SArray<mit_float> * grads) {
  // TODO OpenMP ?
  for (auto i = 0u; i < batch.size; ++i) {
    Gradient(batch[i], preds[i], grads);
  }
} // method Gradient
} // namespace mit
