#include "openmit/framework/worker.h"

namespace mit {

Worker::Worker(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

Worker::~Worker() {
  if (kv_worker_) { delete kv_worker_; kv_worker_ = NULL; }
}

void Worker::Init(const mit::KWArgs & kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  kv_worker_ = new ps::KVWorker<mit_float>(0);
  trainer_.reset(new mit::Trainer(kwargs));
  
  int partid = ps::MyRank();
  int npart = ps::NumWorkers();
  LOG(INFO) << "partid: " << partid << ", npart: " << npart;
  if (cli_param_.task_type == "train") {
    CHECK_NE(cli_param_.train_path, "") << " train_path empty.";
    LOG(INFO) << "load train data: " << cli_param_.train_path;
    train_.reset(new mit::DMatrix(
      cli_param_.train_path, partid, npart, cli_param_.data_format));
    
    CHECK_NE(cli_param_.valid_path, "") << " valid_path empty.";
    LOG(INFO) << "load valid data: " << cli_param_.valid_path;
    valid_.reset(new mit::DMatrix(
      cli_param_.valid_path, partid, npart, cli_param_.data_format));
    
  } else if (cli_param_.task_type == "predict") {
    CHECK_NE(cli_param_.test_path, "") << " test_path empty.";
    test_.reset(new mit::DMatrix(
      cli_param_.test_path, partid, npart, cli_param_.data_format));
  }
  std::string msg = "@w[" + std::to_string(ps::MyRank()) + "] worker init done.";
  LOG(INFO) << msg;
  //LOG(INFO) << "ps worker init done";
}

void Worker::Run() {
  CHECK_GT(cli_param_.batch_size, 0);
  std::vector<float> batch_metric;
  std::string msg = "@w[" + std::to_string(ps::MyRank()) + "] train <epoch, batch, inst>: <";
  for (auto epoch = 1u; epoch <= cli_param_.max_epoch; ++epoch) {
    std::vector<float> train_metric(trainer_->MetricInfo().size(), 0.0);
    int batch_count = 0;
    /* train based on batch data */
    trainer_->timer_stats_->begin(stats.ps_worker_train);
    uint64_t progress = 0u;
    train_->BeforeFirst();
    while (true) {
      trainer_->timer_stats_->begin(stats.ps_worker_io);
      if (! train_->Next()) break;
      auto& block = train_->Value();
      trainer_->timer_stats_->stop(stats.ps_worker_io);
      uint32_t end = 0;
      for (auto i = 0u; i < block.size; i += cli_param_.batch_size) {
        end = i + cli_param_.batch_size >= block.size ? block.size : i + cli_param_.batch_size;
        //if (progress % progress_interval == 0 && cli_param_.is_progress) {
        if (cli_param_.is_progress) {
          LOG(INFO) << msg << epoch << "," << batch_count << "," << progress << ">";
        }
        progress += (end - i);
        if ((end - i) != cli_param_.batch_size && cli_param_.is_progress) {
          LOG(INFO) << msg << epoch << "," << batch_count << "," << progress << ">";
        }
        const auto batch = block.Slice(i, end);
        MiniBatch(batch, batch_metric);

        for (auto i = 0u; i < train_metric.size(); ++i) train_metric[i] += batch_metric[i];
        batch_count += 1;
      }
    } // while 
    trainer_->timer_stats_->stop(stats.ps_worker_train);

    /* metric */
    CHECK(batch_count > 0);
    for (auto& metric : train_metric) metric /= batch_count;
    std::string metric_train_info = MetricMsg(train_metric);
    std::string metric_valid_info = Metric(valid_.get());
    // format: "epoch;train:auc^0.80,logloss^0.1;valid:auc^0.78,logloss^0.11"
    std::string metric_info = std::to_string(epoch);
    metric_info += ";train:" + metric_train_info + ";valid:" + metric_valid_info;
    static_cast<ps::SimpleApp *>(kv_worker_)->Request(mit::signal::METRIC, metric_info, ps::kScheduler);
  } // end for epochs
  
  // send signal to tell server & scheduler worker finish.
  kv_worker_->Wait(kv_worker_->Request(signal::WORKER_FINISH, "worker finish", ps::kScheduler + ps::kServerGroup));

  trainer_->timer_stats_->Print();
  LOG(INFO) << "@worker[" << ps::MyRank() << "] job finish.";
} 

void Worker::MiniBatch(const dmlc::RowBlock<mit_uint>& batch, std::vector<float>& train_metric) {
  /* pull request */
  // sorted unique key 
  trainer_->timer_stats_->begin(stats.ps_worker_pull_dataprepare);
  bool has_extra = cli_param_.data_format == "libfm" && cli_param_.model == "ffm" ? true : false;
  std::vector<ps::Key> keys;
  std::vector<int> extras;
  KeySet(batch, keys, extras, has_extra);
  trainer_->timer_stats_->stop(stats.ps_worker_pull_dataprepare);
  // pull operation 
  trainer_->timer_stats_->begin(stats.ps_worker_pull);
  std::vector<mit_float> weights;
  std::vector<int> lens; 
  kv_worker_->Wait(kv_worker_->Pull(keys, extras, &weights, &lens));
  if (cli_param_.debug) {
    LOG(INFO) << "@w[" << ps::MyRank() << "] pull done. weights from server " << mit::DebugStr<mit_float>(weights.data(), 5);
  }
  trainer_->timer_stats_->stop(stats.ps_worker_pull);
  
  // worker computing 
  std::vector<mit_float> grads(weights.size(), 0.0f);
  trainer_->Run(batch, keys, weights, lens, &grads, train_metric);
  //LOG(INFO) << "minibatch train_metric auc: " << train_metric[0] << ", logloss: " << train_metric[1];

  // push operation (gradient)
  kv_worker_->Push(keys, extras, grads, lens, mit::signal::UPDATE);
}

std::string Worker::Metric(mit::DMatrix* data) {
  std::vector<float> metrics(trainer_->MetricInfo().size(), 0.0f);
  std::vector<float> batch_metric(metrics.size(), 0.0f);
  auto batchsize = cli_param_.batch_size * 10;
  auto num_batch = 0l;
  auto num_inst = 0;
  std::string msg = "@w[" + std::to_string(ps::MyRank()) + "] valid <batch,inst> : <";
  data->BeforeFirst();
  while (true) {
    if (! data->Next()) break;
    auto& block = data->Value();
    uint32_t end = 0;
    for (auto i = 0u; i < block.size; i += batchsize) {
      end = i + batchsize > block.size ? block.size : i + batchsize;
      const auto batch = block.Slice(i, end);
      MetricBatch(batch, batch_metric);
      for (auto idx = 0u; idx < batch_metric.size(); ++idx) {
        metrics[idx] += batch_metric[idx];
      }
      num_batch += 1;
    }
    if (cli_param_.is_progress) {
      num_inst += block.size;
      LOG(INFO) << msg << num_batch << "," << num_inst << ">";
    }
  } // while 

  for (auto& metric_value : metrics) metric_value /= num_batch;
  
  return MetricMsg(metrics);
} // Worker::Metric

std::string Worker::MetricMsg(std::vector<float>& metrics) {
  std::string metric_info("");
  for (auto i = 0u; i < metrics.size(); ++i) {
    metric_info += const_cast<char *>(trainer_->MetricInfo()[i]->Name()) + std::string("^") + std::to_string(metrics[i]);
    if (i != metrics.size() - 1) metric_info += ",";
  }
  //LOG(INFO) << "metric_msg: " << metric_info;
  return metric_info;
} // MetricMsg


void Worker::MetricBatch(const dmlc::RowBlock<mit_uint>& batch, std::vector<float>& metrics_value) {
  bool has_extra = cli_param_.data_format == "libfm" && cli_param_.model == "ffm" ? true : false;
  std::vector<ps::Key> keys;
  std::vector<int> extras;
  KeySet(batch, keys, extras, has_extra);
  
  // pull operation 
  std::vector<mit_float> weights;
  std::vector<int> lens; 
  kv_worker_->Wait(kv_worker_->Pull(keys, extras, &weights, &lens));

  // metric computing 
  metrics_value.clear();
  trainer_->Metric(batch, keys, weights, lens, metrics_value);
} // Worker::MetricBatch

void Worker::KeySet(const dmlc::RowBlock<mit_uint>& batch, 
                    std::vector<mit_uint>& keys, 
                    std::vector<int>& extras, 
                    bool has_extra) {
  keys.clear(); extras.clear();
  if (has_extra) { // deduplicate by map 
    std::map<mit_uint, int> maps;
    for (auto i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
      maps.insert(std::make_pair(batch.index[i], (int)batch.field[i]));
    }
    maps.insert(std::make_pair(0, 0));
    for (auto iter = maps.begin(); iter != maps.end(); ++iter) {
      keys.emplace_back(iter->first);
      extras.emplace_back(iter->second);
    }
  } else if (cli_param_.model == "mf") {
    std::unordered_set<mit_uint> keysets;
    mit_uint index_threashold = ((mit_uint)1 << cli_param_.nbit);
    for (size_t i = 0; i < batch.size; i++) {
      mit_uint userkey = batch.index[batch.offset[i]];
      mit_uint itemkey = batch.index[batch.offset[i] + 1];
      CHECK_LT(userkey, index_threashold);
      CHECK_LT(itemkey, index_threashold); 
      keysets.insert(userkey);
      keysets.insert(NewKey(1, itemkey, cli_param_.nbit));
      if (cli_param_.debug) {
        LOG(INFO) << "user_key:" << userkey << " item_key:" << itemkey << " encode_item_key:" << NewKey(1, itemkey, cli_param_.nbit) << " decode_item_key:" << DecodeField(NewKey(1, itemkey, cli_param_.nbit), cli_param_.nbit);
      }  
    }
    keys.insert(keys.begin(), keysets.begin(), keysets.end());
    std::sort(keys.begin(), keys.end());
  } else { // deduplicate by set
    std::unordered_set<mit_uint> keysets;
    keysets.insert(batch.index + batch.offset[0], batch.index + batch.offset[batch.size]); 
    keysets.insert(0);
    keys.insert(keys.begin(), keysets.begin(), keysets.end());
    std::sort(keys.begin(), keys.end());
  }
} // method Worker::KeySet 

} // namespace mit
