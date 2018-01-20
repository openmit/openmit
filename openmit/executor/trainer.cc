#include "openmit/executor/trainer.h"

namespace mit {

Trainer::Trainer(const mit::KWArgs& kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  // model
  model_ = mit::Model::Create(kwargs);
  // loss
  loss_ = mit::Loss::Create(cli_param_.loss);
  // metric 
  std::vector<std::string> metric_names;
  mit::string::Split(cli_param_.metric, &metric_names, ',');
  CHECK(metric_names.size() > 0) << "metric_names is null. metric: " << cli_param_.metric;
  metrics_.clear();
  for (auto i = 0u; i < metric_names.size(); ++i) {
    mit::Metric* metric = mit::Metric::Create(metric_names[i]);
    CHECK(metric) << "Metric::Create(" << metric_names[i] << ")";
    metrics_.push_back(metric);
  }
  // timer stats 
  timer_stats_ = new mit::TimerStats();
} // Trainer::Trainer

Trainer::~Trainer() {
  if (timer_stats_) { delete timer_stats_; timer_stats_ = NULL; }
}

void Trainer::Run(const dmlc::RowBlock<mit_uint>& batch, std::vector<ps::Key>& keys, std::vector<mit_float>& weights, std::vector<int>& lens, std::vector<mit_float>* grads, std::vector<mit_float>& train_metric) {
  CHECK_EQ(keys.size(), lens.size());
  CHECK_EQ(weights.size(), grads->size());
  size_t nfeature = keys.size();

  /* key -> (offset, count) */
  timer_stats_->begin(stats.ps_worker_map_prepare);
  std::unordered_map<mit_uint, std::pair<size_t, int> > key2offset;
  size_t offset = 0;
  std::string str = "";
  for (size_t i = 0; i < nfeature; ++i) {
    key2offset[keys[i]] = std::make_pair(offset, lens[i]);
    offset += lens[i];
  }
  timer_stats_->stop(stats.ps_worker_map_prepare);
  
  /* predict model expression score based on batch data */
  timer_stats_->begin(stats.ps_worker_model_predict);
  std::vector<mit_float> preds(batch.size, 0.0);
  model_->Predict(batch, weights, key2offset, preds);
  if (cli_param_.debug) {
    LOG(INFO) << "trainer model predict " << mit::DebugStr<mit_float>(preds.data(), 5);
  }
  timer_stats_->stop(stats.ps_worker_model_predict);
  
  /* gradient computing */
  std::vector<mit_float> loss_grads(batch.size, 0.0);
  auto nthread = cli_param_.num_thread; 
  int chunksize = batch.size / nthread;
  chunksize = batch.size % nthread == 0 ? chunksize : chunksize + 1;
  // gradient for loss 
  timer_stats_->begin(stats.ps_worker_calc_loss);
  #pragma omp parallel for num_threads(nthread)
  for (auto i = 0u; i < batch.size; ++i) {
    loss_grads[i] = loss_->gradient(batch[i].get_label(), preds[i]);
  }
  timer_stats_->stop(stats.ps_worker_calc_loss);
  // gradient for model
  timer_stats_->begin(stats.ps_worker_model_gradient);
  model_->Gradient(batch, weights, key2offset, loss_grads, grads);
  if (cli_param_.debug) {
    LOG(INFO) << "trainer model gradient " << mit::DebugStr<mit_float>(grads->data(), 5);
  }
  timer_stats_->stop(stats.ps_worker_model_gradient);
  
  /* metric train based on batch data */
  timer_stats_->begin(stats.ps_worker_train_metric);
  train_metric.resize(metrics_.size(), 0.0);
  if (cli_param_.is_train_metric) {
    std::vector<mit_float> labels(batch.label, batch.label + batch.size);
    for (auto i = 0u; i < train_metric.size(); ++i) {
      train_metric[i] = metrics_[i]->Eval(preds, labels);
    }
  }
  timer_stats_->stop(stats.ps_worker_train_metric);
} // Trainer::Run

void Trainer::Metric(const dmlc::RowBlock<mit_uint>& batch, std::vector<ps::Key>& keys, std::vector<mit_float>& weights, std::vector<int>& lens, std::vector<float>& metrics_value) {
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
  if (cli_param_.objective != "regression") {
    #pragma omp parallel for num_threads(cli_param_.num_thread)
    for (auto i = 0u; i < batch.size; ++i) {
      preds[i] = mit::math::sigmoid(preds[i]);
    }
  }
  std::vector<mit_float> labels(batch.label, batch.label + batch.size);
  
  // metric 
  auto num_metric = metrics_.size();
  metrics_value.resize(num_metric, 0.0f);
  for (auto i = 0u; i < num_metric; ++i) {
    float value = metrics_[i]->Eval(preds, labels);
    metrics_value[i] = value;
  }
} // Trainer::Metric

void Trainer::Loss(const dmlc::RowBlock<mit_uint>& batch, const std::vector<mit_float>* predict, std::vector<mit_float>* loss) {
  // TODO loss_->Loss()
}

} // namespace mit
