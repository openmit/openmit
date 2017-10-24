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
    return mit::LR::Get(kwargs);
    //return mit::FM::Get(kwargs);
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

// implementation of gradient based on batch instance for ps
void Model::Gradient(const dmlc::RowBlock<mit_uint> & batch, 
                     const std::vector<mit_float> & weights, 
                     key2offset_type & key2offset,
                     const std::vector<mit_float> & preds, 
                     std::vector<mit_float> * grads) {
  CHECK_EQ(batch.size, preds.size()) << "block.size != preds.size()";
  CHECK_EQ(weights.size(), grads->size()) << "weights.size() != grads.size()";
  // \sum grad for w and v
  for (auto i = 0u; i < batch.size; ++i) {
    Gradient(batch[i], weights, key2offset, preds[i], grads);
  }

  // \frac{1}{batch.size} \sum grad 
  for (auto i = 0u; i < grads->size(); ++i) {
    (*grads)[i] /= batch.size;
  }
}

void Model::Gradient(const dmlc::RowBlock<mit_uint> & batch, 
                     std::vector<mit_float> & preds, 
                     mit::SArray<mit_float> * grads) {
  CHECK_EQ(batch.size, preds.size());
  for (auto i = 0u; i < batch.size; ++i) {
    Gradient(batch[i], preds[i], grads);
  }
  if (batch.size == 1) return ;
  CHECK(batch.size > 0) << "batch.size <= 0 in Model::Gradient";
  for (auto j = 0u; j < grads->size(); ++j) {
    (*grads)[j] /= batch.size;
  }
} // method Gradient 

} // namespace mit
