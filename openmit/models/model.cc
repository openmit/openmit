//#include "openmit/models/factorization_machine.h"
//#include "openmit/models/fieldaware_factorization_machine.h"
#include "openmit/models/logistic_regression.h"
#include "openmit/models/model.h"

namespace mit {

Model * Model::Create(const mit::KWArgs & kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  if (cli_param_.model == "lr") {
    return mit::LR::Get(kwargs);
  } else if (cli_param_.model == "fm") {
    return mit::LR::Get(kwargs);
    //return mit::FM::Get(kwargs);
  } else if (cli_param_.model == "ffm") {
    return mit::LR::Get(kwargs);
    //return mit::FFM::Get(kwargs);
  } else {
    LOG(ERROR) <<
      "model not in [lr, fm, ffm], model: " << cli_param_.model;
    return nullptr;
  }
}

void Model::Predict(const dmlc::RowBlock<mit_uint> & batch,
                    const std::vector<mit_float> & weights, 
                    std::unordered_map<mit_uint, std::pair<size_t, int> > & key2offset, 
                    std::vector<mit_float> & preds, 
                    bool is_norm) {
  CHECK_EQ(batch.size, preds.size());
  // TODO OpenMP?
  for (auto i = 0u; i < batch.size; ++i) {
    preds[i] = Predict(batch[i], weights, key2offset, is_norm);
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
void Model::Gradient(const dmlc::RowBlock<mit_uint> & batch, const std::vector<mit_float> & weights, std::unordered_map<mit_uint, std::pair<size_t, int> > & key2offset, const std::vector<mit_float> & preds, std::vector<mit_float> * grads) {
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
                    std::vector<mit_float> * preds, 
                    bool is_norm) {
  CHECK_EQ(batch.size, preds->size());
  // TODO OpenMP?
  for (auto i = 0u; i < batch.size; ++i) {
    (*preds)[i] = Predict(batch[i], weight, is_norm);
  }
} // method Predict

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
