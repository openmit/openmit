#include <cmath>
#include "rabit/rabit.h"
#include "openmit/framework/admm/mpi_worker.h"

namespace mit {

MPIWorker::MPIWorker(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

MPIWorker::~MPIWorker() {
  weight_.clear(); delete[] weight_.data();
  dual_.clear(); delete[] dual_.data();
  // TODO
}

void MPIWorker::Init(const mit::KWArgs & kwargs) {
  // 1. initialize parameter
  cli_param_.InitAllowUnknown(kwargs);
  admm_param_.InitAllowUnknown(kwargs);

  // 2. data
  int partid = rabit::GetRank();
  int npart = rabit::GetWorldSize();
  if (cli_param_.task == "train") {
    CHECK_NE(cli_param_.train_path, "")
      << "train_path should not be empty for train.";
    train_.reset(new mit::DMatrix(
      cli_param_.train_path, partid, npart, cli_param_.data_format));
    CHECK_NE(cli_param_.valid_path, "")
      << "valid_path should not be empty for train.";
    valid_.reset(new mit::DMatrix(
      cli_param_.valid_path, partid, npart, cli_param_.data_format));
  } else if (cli_param_.task == "predict") {
    CHECK_NE(cli_param_.test_path, "")
      << "test path should not be empty for predict task.";
    test_.reset(new mit::DMatrix(
      cli_param_.test_path, partid, npart, cli_param_.data_format));
  } else {
    LOG(ERROR) << "error mpi_worker init.";
  }

  // 3. model & optimizer
  if (cli_param_.task == "train") {
    // ldim: worker local partial-data max dimension 
    uint32_t ldim = std::max(train_->NumCol(), valid_->NumCol());
    std::vector<uint32_t> dim(1, ldim);
    rabit::Allreduce<rabit::op::Max>(&dim[0], dim.size());
    rabit::Broadcast(dim.data(), sizeof(uint32_t) * dim.size(), 0);

    uint32_t max_dim = dim[0];
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf(
      "[INFO] @node[0] global max feature dim: %d\n", max_dim);
    }
    weight_.resize(max_dim + 1, 0.0f);
    dual_.resize(max_dim + 1, 0.0f);

    opt_.reset(mit::Opt::Create(kwargs, cli_param_.optimizer));
    opt_->Init(max_dim);

    LOG(INFO) << "worker init done...";
  } if (cli_param_.task == "predict") {
    // Load Model for prediction
  } else {
    // TODO
  }
  // initialize model 
  model_.reset(mit::Model::Create(kwargs));
}

void MPIWorker::Run(mit_float * global, const size_t size) {
  auto lsize = weight_.size();
  CHECK_EQ(size, lsize) 
    << "global_model.size != local_model.size";
  weight_.clear(); 
  weight_.CopyFrom(global, size);

  train_->BeforeFirst();
  while (train_->Next()) {
    auto & block = train_->Value();
    // TODO optimized to matrix/vector computation
    size_t end = 0;
    if (block.size <= cli_param_.batch_size) {
      MiniBatch(block);
    } else {
      for (auto i = 0u; i < block.size; i += cli_param_.batch_size) {
        end = i + cli_param_.batch_size >= block.size ? 
          block.size : i + cli_param_.batch_size;
        const auto batch = block.Slice(i, end);
        MiniBatch(batch);
      }
    }
  } // while
}

void MPIWorker::MiniBatch(const dmlc::RowBlock<mit_uint> & batch) {
  // prediction: batch 
  mit::SArray<mit_float> preds(batch.size, 0.0);
  model_->Predict(batch, weight_, &preds, true);
  // optimizer : gradient
  mit::SArray<mit_float> grads(weight_.size(), 0.0);
  model_->Gradient(batch, preds, &grads);
  // optimizer : weight_
  opt_->Run(grads, &weight_);
}

// dual_[j] <-- dual_[j] + \rho * (w_[j] - \theta[j])
void MPIWorker::UpdateDual(mit_float * global, const size_t size) {
  auto dsize = dual_.size();
  CHECK_EQ(size, dsize) 
    << "global_model.size != dual.size";
  for (auto j = 0u; j < size; ++j) {
    dual_[j] += admm_param_.rho * (weight_[j] - *(global + j));
  }
}

void MPIWorker::Debug() {
  DebugWeight();
  DebugDual();
}

void MPIWorker::DebugWeight() {
  LOG(INFO) << mit::DebugStr(weight_.data(), weight_.size());
} // method DebugWeight

void MPIWorker::DebugDual() {
  LOG(INFO) << mit::DebugStr(dual_.data(), dual_.size());
} // method DebugWeight

} // namespace mit
