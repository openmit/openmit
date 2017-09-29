#include "openmit/models/factorization_machine.h"
#include "openmit/models/fieldaware_factorization_machine.h"
#include "openmit/models/logistic_regression.h"
#include "openmit/models/model.h"

namespace mit {

DMLC_REGISTER_PARAMETER(ModelParam);

Model * Model::Create(const mit::KWArgs & kwargs) {
  ModelParam tmp_param_;
  tmp_param_.InitAllowUnknown(kwargs);
  if (tmp_param_.model_type == "lr") {
    return mit::LR::Get(kwargs);
  } else if (tmp_param_.model_type == "fm") {
    return mit::FM::Get(kwargs);
  } else if (tmp_param_.model_type == "ffm") {
    return mit::FFM::Get(kwargs);
  } else {
    LOG(ERROR) <<
      "model_type not in [lr, fm, ffm], model_type: " << tmp_param_.model_type;
    return nullptr;
  }
}

// implementation of prediction based on batch instance for ps
void Model::Predict(const dmlc::RowBlock<mit_uint> & row_block,
                    mit::PMAPT & weight,
                    std::vector<mit_float> * preds,
                    bool is_norm) {
  CHECK_EQ(row_block.size, preds->size());
  // TODO OpenMP?
  for (auto i = 0u; i < row_block.size; ++i) {
    (*preds)[i] = Predict(row_block[i], weight, is_norm);
  }
}

// implementation of gradient based on batch instance for ps
void Model::Gradient(
    const dmlc::RowBlock<mit_uint> & block,
    std::vector<mit_float> & preds,
    PMAPT & weight,
    PMAPT * grad) {

  CHECK_EQ(block.size,  preds.size());
  CHECK_EQ(weight.size(), grad->size());
  // \sum grad
  for (auto i = 0u; i < block.size; ++i) {
    Gradient(block[i], preds[i], weight, grad);
  }
  // \frac{1}{block.size} \sum grad
  for (auto kunit : *grad) {
    auto feati = kunit.first;
    auto batch_grad = 1.0 * (*grad)[feati]->Get(0) / block.size;
    (*grad)[feati]->Set(0, batch_grad);
  }
} // method Gradient

void Model::Predict(const dmlc::RowBlock<mit_uint> & batch, 
                    mit::SArray<mit_float> & weight, 
                    mit::SArray<mit_float> * preds, 
                    bool is_norm) {
  CHECK_EQ(batch.size, preds->size());
  // TODO OpenMP?
  for (auto i = 0u; i < batch.size; ++i) {
    (*preds)[i] = Predict(batch[i], weight, is_norm);
  }
} // method Predict

void Model::Gradient(const dmlc::RowBlock<mit_uint> & batch, 
                     mit::SArray<mit_float> & preds, 
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
