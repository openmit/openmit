#include "openmit/model/fm.h"
#include "openmit/model/ffm.h"
#include "openmit/model/linear_reg.h"
#include "openmit/model/model.h"

namespace mit {

Model::Model(const mit::KWArgs & kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  model_param_.InitAllowUnknown(kwargs);
  entry_meta_.reset(new mit::EntryMeta(model_param_));
  random_.reset(mit::math::ProbDistr::Create(model_param_));
  optimizer_.reset(mit::Optimizer::Create(kwargs));
}

Model::~Model() {}

Model* Model::Create(const mit::KWArgs& kwargs) {
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
    LOG(FATAL) << "unknown model. " << model;
    return nullptr;
  }
}

void Model::Gradient(const dmlc::RowBlock<mit_uint> & batch, std::vector<mit_float> & preds, mit::SArray<mit_float> * grads) {
  // TODO OpenMP ?
  for (auto i = 0u; i < batch.size; ++i) {
    Gradient(batch[i], preds[i], grads);
  }
} // Model::Gradient

void Model::Predict(const dmlc::RowBlock<mit_uint>& batch, mit::SArray<mit_float>& weight, std::vector<mit_float>* preds, bool norm) {
  // TODO
} // Model::Predict

} // namespace mit
