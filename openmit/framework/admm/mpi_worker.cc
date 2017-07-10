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
  param_.InitAllowUnknown(kwargs);
  admm_param_.InitAllowUnknown(kwargs);

  // 2. data
  int partid = rabit::GetRank();
  int npart = rabit::GetWorldSize();
  if (param_.task == "train") {
    CHECK_NE(param_.train_path, "")
      << "train_path should not be empty for train.";
    train_.reset(new mit::DMatrix(
      param_.train_path, partid, npart, param_.data_format));
    CHECK_NE(param_.valid_path, "")
      << "valid_path should not be empty for train.";
    valid_.reset(new mit::DMatrix(
      param_.valid_path, partid, npart, param_.data_format));
  } else if (param_.task == "predict") {
    CHECK_NE(param_.test_path, "")
      << "test path should not be empty for predict task.";
    test_.reset(new mit::DMatrix(
      param_.test_path, partid, npart, param_.data_format));
  } else {
    LOG(ERROR) << "error mpi_worker init.";
  }

  if (param_.task == "train") {
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

    // 3. model & optimizer
    opt_.reset(mit::Opt::Create(kwargs, param_.optimizer));
    LOG(INFO) << "worker init done...";
  } if (param_.task == "predict") {
    // Load Model for prediction
  } else {
    // TODO
  }
  model_.reset(mit::Model::Create(kwargs));
}

void MPIWorker::Update(mit_float * global, const size_t size) {
  auto lsize = weight_.size();
  CHECK_EQ(size, lsize) 
    << "global_model.size != local_model.size";
  weight_.clear(); weight_.CopyFrom(global, size);

  train_->BeforeFirst();
  while (train_->Next()) {
    auto & block = train_->Value();
    for (auto i = 0u; i < block.size; ++i) {
      //auto pred = model_->Predict(block[i], weight_);
      //opt_->Update(block[i], pred, weight_);
    }
  }
}

void MPIWorker::UpdateDual(mit_float * global, const size_t size) {
  auto dsize = dual_.size();
  CHECK_EQ(size, dsize) 
    << "global_model.size != dual.size";
  for (auto j = 0u; j < size; ++j) {
    dual_[j] += admm_param_.rho * (weight_[j] - *(global + j));
  }
}

} // namespace mit
