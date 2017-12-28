#include "openmit/executor/trainer.h"

namespace mit {

Trainer::Trainer(const mit::KWArgs & kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  // model
  model_ = mit::PSModel::Create(kwargs);
  // loss
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
void Trainer::Run(std::unordered_map<ps::Key, mit::mit_float>& rating_map,
                  std::vector<ps::Key>& user_keys,
                  std::vector<mit_float> & user_weights,
                  std::vector<int> & user_lens,
                  std::vector<ps::Key> & item_keys,
                  std::vector<mit_float> & item_weights,
                  std::vector<int> & item_lens,
                  std::vector<mit_float> * user_grads,
                  std::vector<mit_float> * item_grads) {
  CHECK_EQ(user_keys.size(), user_lens.size());
  CHECK_EQ(user_weights.size(), user_grads->size());
  CHECK_EQ(item_keys.size(), item_lens.size());
  CHECK_EQ(item_weights.size(), item_grads->size());
  if (cli_param_.optimizer=="als") { //matrix fatorization by als
    if (cli_param_.debug) {
      LOG(INFO) << "factorization by als begin!";
    }
    model_->SolveByAls(rating_map,
                       user_keys, 
                       user_weights,
                       user_lens,
                       item_keys,
                       item_weights,
                       item_lens,
                       user_grads,
                       item_grads);
    if (cli_param_.debug) {
      LOG(INFO) << "factorization by als completed!";
    }
    return;
  }
  else { //matrix fatorization by sgd
    auto user_feature_size = user_keys.size();
    auto item_feature_size = item_keys.size();
    size_t user_offset = 0;
    size_t item_offset = 0;
    for (auto i = 0u; i < user_feature_size; i++){
      mit_uint user_id = user_keys[i];
      mit_uint user_len = user_lens[i];
      item_offset = 0;
      for (auto j = 0u; j < item_feature_size; j++){
        mit_uint item_id = item_keys[j];
        mit_uint item_len = item_lens[j];
        CHECK_EQ(user_len, item_len);
        mit_uint new_key = mit::NewKey(
          user_id, item_id, cli_param_.nbit);
        if (cli_param_.debug) {
          //LOG(INFO) << "user_id:" << user_id << " item_id:" << item_id << " newkey:" << new_key
         //          << " decode:" << DecodeFeature(new_key, cli_param_.nbit)<<" "<< DecodeField(new_key, cli_param_.nbit);
        }
        if (rating_map.find(new_key) == rating_map.end()){
          item_offset += item_len;
          continue;
        }
  
        auto mfunc_value = model_->Predict(user_weights, user_offset,
                                           item_weights, item_offset, user_len);
        auto lossgrad_value = loss_->gradient(rating_map[new_key], mfunc_value);
        if (cli_param_.debug) {
          //LOG(INFO) << "rating:" << rating_map[new_key];
          //LOG(INFO) << "mfunc_value:" << mfunc_value;
          //LOG(INFO) << "lossgrad_value:" << lossgrad_value;
          //LOG(INFO) << "user gradient before update:" << mit::DebugStr(user_grads->data(), user_grads->size(), 15);
          ///LOG(INFO) << "item gradient before update:" << mit::DebugStr(item_grads->data(), item_grads->size(), 12);
        }
        model_->Gradient(lossgrad_value, user_weights, user_offset,
                         item_weights, item_offset, user_len,
                         user_grads, item_grads);
        if (cli_param_.debug) {
          //LOG(INFO) << "user gradient after update:" << mit::DebugStr(user_grads->data(), user_grads->size(), 15);
          //LOG(INFO) << "item gradient after update:" << mit::DebugStr(item_grads->data(), item_grads->size(), 12);
        }
        item_offset += item_len;
      }
      user_offset += user_len;
    }
  }//else sgd
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
    std::string msg("metric preds: [");
    msg += std::to_string(preds.size()) + "] " + mit::DebugStr<mit_float>(preds.data(), 10, 10);
    msg += "\nmetric label: [";
    msg += std::to_string(labels.size()) + "] " + mit::DebugStr<mit_float>(labels.data(), 10, 10);
    LOG(INFO) << msg;
  }
  // metric 
  auto metric_count = metrics_.size();
  metrics_value.resize(metric_count, 0.0f);
  for (auto i = 0u; i < metric_count; ++i) {
    float value = metrics_[i]->Eval(preds, labels);
    metrics_value[i] = value;
  }
}

void Trainer::Metric(std::unordered_map<ps::Key, mit::mit_float>& rating_map,
                     std::vector<ps::Key> & user_keys,
                     std::vector<mit_float> & user_weights,
                     std::vector<int> & user_lens,
                     std::vector<ps::Key> & item_keys,
                     std::vector<mit_float> & item_weights,
                     std::vector<int> & item_lens,
                     std::vector<float> & metrics_value) {
  CHECK_EQ(user_keys.size(), user_lens.size());
  CHECK_EQ(item_keys.size(), item_lens.size());

  auto user_feature_size = user_keys.size();
  auto item_feature_size = item_keys.size();
  size_t user_offset = 0;
  size_t item_offset = 0;
  std::vector<mit_float> preds;
  std::vector<mit_float> labels;
  for (auto i = 0u; i < user_feature_size; i++){
    mit_uint user_id = user_keys[i];
    mit_uint user_len = user_lens[i];
    item_offset = 0;
    for (auto j = 0u; j < item_feature_size; j++){
      mit_uint item_id = item_keys[j];
      mit_uint item_len = item_lens[j];
      CHECK_EQ(user_len, item_len);
      mit_uint new_key = mit::NewKey(
        user_id, item_id, cli_param_.nbit);
      if (rating_map.find(new_key) == rating_map.end()){
        item_offset += item_len;
        continue;
      }
      auto mfunc_value = model_->Predict(user_weights, user_offset,
                                         item_weights, item_offset, user_len);
      preds.push_back(mfunc_value);
      labels.push_back(rating_map[new_key]);
      item_offset += item_len;
    }
    user_offset += user_len;
  }
  if (cli_param_.debug) {
    LOG(INFO) << "metric preds: [" << preds.size() << "] "
      << mit::DebugStr<mit_float>(preds.data(), preds.size(), 10);
    LOG(INFO) << "metric labels: " << labels.size() << "] "
      << mit::DebugStr<mit_float>(labels.data(),labels.size(), 10);
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
