#include <cmath>
#include "rabit/rabit.h"
#include "openmit/framework/mpi/worker.h"
#include "openmit/tools/dstruct/dstring.h"
#include "openmit/tools/util/type_conversion.h"

namespace mit {

MPIWorker::MPIWorker(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

MPIWorker::~MPIWorker() {
  weight_.clear(); delete[] weight_.data();
  dual_.clear(); delete[] dual_.data();
  // TODO
  // metric 
  if (! metrics_.empty()) {
    for (auto i = 0u; i < metrics_.size(); ++i) {
      if (!metrics_[i]) continue;
      delete metrics_[i]; metrics_[i] = nullptr;
    }
  }
}

void MPIWorker::Init(const mit::KWArgs & kwargs) {
  // 1. initialize parameter
  cli_param_.InitAllowUnknown(kwargs);
  admm_param_.InitAllowUnknown(kwargs);

  // 2. data
  int partid = rabit::GetRank();
  int npart = rabit::GetWorldSize();
  if (cli_param_.task_type == "train") {
    CHECK_NE(cli_param_.train_path, "")
      << "train_path should not be empty for train.";
    train_.reset(new mit::DMatrix(
      cli_param_.train_path, partid, npart, cli_param_.data_format));

    size_t train_inst = 0;
    train_->BeforeFirst();
    while (train_->Next()) {
      auto block = train_->Value();
      train_inst += block.size;
    }
    
    CHECK_NE(cli_param_.valid_path, "")
      << "valid_path should not be empty for train.";
    valid_.reset(new mit::DMatrix(
      cli_param_.valid_path, partid, npart, cli_param_.data_format));
  
  } else if (cli_param_.task_type == "predict") {
    CHECK_NE(cli_param_.test_path, "")
      << "test path should not be empty for predict task.";
    test_.reset(new mit::DMatrix(
      cli_param_.test_path, partid, npart, cli_param_.data_format));
  } else {
    LOG(ERROR) << "error mpi_worker init.";
  }

  // 3. model && optimizer && metric
  if (cli_param_.task_type == "train") {
    // ldim: worker local partial-data max dimension 
    //uint32_t ldim = std::max(train_->NumCol(), valid_->NumCol());
    uint32_t ldim = 1e8; // TODO
    std::vector<uint32_t> dim(1, ldim);
    rabit::Allreduce<rabit::op::Max>(&dim[0], dim.size());
    rabit::Broadcast(dim.data(), sizeof(uint32_t) * dim.size(), 0);
    uint32_t max_dim = dim[0];
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("global max feature dim: %d\n", max_dim);
    }
    weight_.resize(max_dim + 1, 0.0f);
    dual_.resize(max_dim + 1, 0.0f);
    // optimizer
    optimizer_.reset(mit::Optimizer::Create(kwargs));
    optimizer_->Init(max_dim);
    LOG(INFO) << "@worker[" <<  rabit::GetRank() 
      << "] mpiworker init done for train task.";
  } else if (cli_param_.task_type == "predict") {
    // TODO Load Model for prediction
  } else {
    LOG(ERROR) << "task_type not in [train, predict].";
  }
  // model  
  model_.reset(mit::Model::Create(kwargs));
  // metric 
  std::vector<std::string> metric_names;
  mit::string::Split(cli_param_.metric, &metric_names, ',');
  CHECK(metric_names.size() > 0) 
    << "metric_names is null. metric: " << cli_param_.metric;
  metrics_.clear();
  for (auto i = 0u; i < metric_names.size(); ++i) {
    mit::Metric * metric = mit::Metric::Create(metric_names[i]);
    CHECK(metric) << "Metric::Create(" << metric_names[i] << ")";
    metrics_.push_back(metric);
  }
}

void MPIWorker::Run(mit_float * global, 
                    const size_t size, 
                    const size_t epoch) {
  auto lsize = weight_.size();
  CHECK_EQ(size, lsize) 
    << "global_model.size != local_model.size";
  weight_.clear(); 
  weight_.CopyFrom(global, size);

  uint64_t progress = 0u;
  CHECK(cli_param_.job_progress > 0) << "parameter job_progress > 0";
  size_t progress_interval = cli_param_.batch_size * cli_param_.job_progress;
  train_->BeforeFirst();
  while (train_->Next()) {
    auto & block = train_->Value();
    // TODO optimized to matrix/vector computation
    uint32_t end = 0;
    for (auto i = 0u; i < block.size; i += cli_param_.batch_size) {
      end = i + cli_param_.batch_size >= block.size ? 
        block.size : i + cli_param_.batch_size;
      if (progress % progress_interval == 0 && cli_param_.is_progress) {
        LOG(INFO) << "@worker[" << rabit::GetRank() << "] progress \
                  <epoch, inst>: <" << epoch << ", " << progress << ">";
      }
      progress += (end - i);
      if ((end - i) != cli_param_.batch_size && cli_param_.is_progress) {
        LOG(INFO) << "@worker[" << rabit::GetRank() << "] progress \
                  <epoch, inst>: <" << epoch << ", " << end << ">";
      }
      const auto batch = block.Slice(i, end);
      MiniBatch(batch);
    }
  } // while
}

void MPIWorker::MiniBatch(const dmlc::RowBlock<mit_uint> & batch) {
  // prediction: batch 
  std::vector<mit_float> preds(batch.size, 0.0f);
  model_->Predict(batch, weight_, &preds);
  // optimizer : gradient
  mit::SArray<mit_float> grads(weight_.size(), 0.0f);
  model_->Gradient(batch, preds, &grads);
  // optimizer : weight_
  optimizer_->Run(grads, &weight_);
}

// dual_[j] <-- dual_[j] + \rho * (w_[j] - \theta[j])
void MPIWorker::UpdateDual(mit_float * global, const size_t size) {
  auto dsize = dual_.size();
  CHECK_EQ(size, dsize) << "global_model.size != dual.size";
  for (auto j = 0u; j < size; ++j) {
    dual_[j] += admm_param_.rho * (weight_[j] - *(global + j));
  }
}

std::string MPIWorker::Metric(const std::string & data_type, 
                              mit_float * global, 
                              const size_t size) {
  // predict
  std::vector<float> preds;
  std::vector<float> labels;
  if (data_type == "train") {
    MetricPredict(train_.get(), global, size, preds, labels);
  } else if (data_type == "valid") {
    MetricPredict(valid_.get(), global, size, preds, labels);
  } else if (data_type == "test") {
    MetricPredict(test_.get(), global, size, preds, labels);
  } else {
    LOG(ERROR) << "data_type not in [train, valid, test]";
  }
  // metric computing
  auto metric_count = metrics_.size();
  std::vector<float> metric_partial(metric_count, 0.0f);
  for (auto i = 0u; i < metric_count; ++i) {
    float value = metrics_[i]->Eval(preds, labels);
    metric_partial[i] = value;
  }
  // allreduce and broadcast 
  rabit::Allreduce<rabit::op::Sum>(metric_partial.data(), metric_count);
  for (auto i = 0u; i < metric_partial.size(); ++i) {
    metric_partial[i] /= rabit::GetWorldSize();
  }
  rabit::Broadcast(metric_partial.data(), sizeof(mit_float) * metric_count, 0);

  // metric result
  std::string result;
  for (auto i = 0u; i < metrics_.size(); ++i) {
    result += std::string(metrics_[i]->Name());
    result += std::string(": ") + mit::NumToString<float>(metric_partial[i]);
    if (i != metrics_.size() - 1) result += ", ";
  }
  return result;
}

void MPIWorker::MetricPredict(mit::DMatrix* data, 
                              mit_float* global, 
                              const size_t& size, 
                              std::vector<mit_float>& preds, 
                              std::vector<mit_float>& labels) {
  mit::SArray<mit_float> global_model(global, size);
  std::vector<mit_float> preds_tmp;
  data->BeforeFirst();
  while (data->Next()) {
    const auto& block = data->Value();
    labels.insert(labels.end(), block.label, block.label + block.size);

    preds_tmp.clear(); preds_tmp.resize(block.size, 0.0f);
    model_->Predict(block, global_model, &preds_tmp);
    preds.insert(preds.end(), preds_tmp.begin(), preds_tmp.end());
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
