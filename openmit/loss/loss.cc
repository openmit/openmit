#include "openmit/loss/loss.h"

namespace mit {

// implementation of squared-loss function
mit_float SquaredLoss::
LossFunc(const mit_float & pred, const mit_float & label) {
  auto loss_value = (label - pred) * (label - pred);
  return loss_value;
}

// implementation of logit-loss function
mit_float LogitLoss::
LossFunc(const mit_float & pred, const mit_float & label) {
  auto loss_value = label * log(pred) + (1 - label) * log(1 - pred);
  return loss_value;
}

// implementation of calculate loss based on a batch data
mit_float Loss::CalcLoss(const dmlc::RowBlock<mit_uint> & row_block, 
                         const std::vector<mit_float> & pred_value) {
  CHECK_EQ(row_block.size, pred_value.size());
  mit_float loss = 0;
  for (auto i = 0u; i < pred_value.size(); ++i) {
    loss += LossFunc(pred_value[i], row_block.label[i]); 
  }
  return loss;
}

// create a loss function
Loss * Loss::Create(std::string loss_type) {
  if (loss_type == "squared") {
    return SquaredLoss::Get();
  } else if (loss_type == "logit") {
    return LogitLoss::Get();
  } else {
    LOG(ERROR) << "loss_type not recognized. loss_type: " << loss_type;
    return nullptr;
  }
}

} // namespace mit
