#include <cmath>

#include "rabit/rabit.h"

#include "openmit/framework/admm/mpi_worker.h"


namespace mit {

void MPIWorker::Init(const mit::KWArgs & kwargs) {
  // 1. initialize parameter
  param_.InitAllowUnknown(kwargs);

  // 2. data
  int partid = rabit::GetRank();
  int npart = rabit::GetWorldSize();
  if (param_.task == "train") {
    CHECK_NE(param_.train_path, "")
      << "train_path should not be empty for train.";
    train_set_.reset(new mit::DMatrix(
      param_.train_path, partid, npart, param_.data_format));
    CHECK_NE(param_.valid_path, "")
      << "valid_path should not be empty for train.";
    valid_set_.reset(new mit::DMatrix(
      param_.valid_path, partid, npart, param_.data_format));
  }
  uint32_t local_dim = std::max(train_set_->NumCol(), valid_set_->NumCol());
  std::vector<uint32_t> dim(1, local_dim);
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
}
} // namespace mit
