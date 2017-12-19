#include "openmit/executor/trainer.h"

namespace mit {

Trainer::Trainer(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

void Trainer::Init(const mit::KWArgs & kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  model_ = mit::Model::Create(kwargs);
  loss_ = mit::Loss::Create(cli_param_.loss);
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

void Trainer::Run(const dmlc::RowBlock<mit_uint> & batch, std::vector<ps::Key> & keys, std::vector<mit_float> & weights, std::vector<int> & lens, std::vector<mit_float> * grads) {
  CHECK_EQ(keys.size(), lens.size());
  CHECK_EQ(weights.size(), grads->size());

  size_t nfeature = keys.size();
  // key -> (offset, count) 
  std::unordered_map<mit_uint, std::pair<size_t, int> > key2offset;
  size_t offset = 0;
  std::string str = "";
  for (size_t i = 0; i < nfeature; ++i) {
    key2offset[keys[i]] = std::make_pair(offset, lens[i]);
    offset += lens[i];
  }
  // predict based on batch data
  std::vector<mit_float> preds(batch.size, 0.0);
  model_->Predict(batch, weights, key2offset, preds, false);
  
  // gradient computing
  std::vector<mit_float> loss_grads(batch.size, 0.0);
  auto num_thread = cli_param_.num_thread;
  int chunksize = batch.size / num_thread;
  chunksize = batch.size % num_thread == 0 ? chunksize : chunksize + 1;
  #pragma omp parallel for num_threads(num_thread)
  for (auto i = 0u; i < batch.size; ++i) {
    loss_grads[i] = loss_->gradient(batch[i].get_label(), preds[i]);
  }
  model_->Gradient(batch, weights, key2offset, loss_grads, grads);

  if (cli_param_.debug) {
    LOG(INFO) << "Trainer::Run grads: [" << grads->size() << "] "
      << mit::DebugStr<mit_float>(grads->data(), 10, 10);
  }
}

void Trainer::Metric(const dmlc::RowBlock<mit_uint> & batch, std::vector<ps::Key> & keys, std::vector<mit_float> & weights, std::vector<int> & lens, std::vector<float> & metrics_value) {
  CHECK_EQ(keys.size(), lens.size());
  size_t nfeature = keys.size();
  // key --> (offset, count)
  std::unordered_map<mit_uint, std::pair<size_t, int> > key2offset;
  size_t offset = 0;
  for (size_t i = 0; i < nfeature; ++i) {
    key2offset[keys[i]] = std::make_pair(offset, lens[i]);
    offset += lens[i];
  }

  // predict 
  std::vector<mit_float> preds(batch.size);
  model_->Predict(batch, weights, key2offset, preds);
  std::vector<mit_float> labels(batch.label, batch.label + batch.size);
  if (cli_param_.debug) {
    LOG(INFO) << "metric preds: [" << preds.size() << "] "
      << mit::DebugStr<mit_float>(preds.data(), 10, 10);
    LOG(INFO) << "metric labels: " << labels.size() << "] "
      << mit::DebugStr<mit_float>(labels.data(), 10, 10);
  }
  // metric 
  auto metric_count = metrics_.size();
  metrics_value.resize(metric_count, 0.0f);
  for (auto i = 0u; i < metric_count; ++i) {
    float value = metrics_[i]->Eval(preds, labels);
    metrics_value[i] = value;
  }
}

void Trainer::Loss(
    const dmlc::RowBlock<mit_uint> & batch, 
    const std::vector<mit_float> * predict, 
    std::vector<mit_float> * loss) {
  // TODO loss_->Loss()
}

} // namespace mit