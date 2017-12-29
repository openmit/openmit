#include "openmit/framework/ps/worker.h"

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
  model_.reset(mit::PSModel::Create(kwargs));
  
  int partid = ps::MyRank();
  int npart = ps::NumWorkers();
  LOG(INFO) << "partid: " << partid << ", npart: " << npart;
  if (cli_param_.task_type == "train") {
    CHECK_NE(cli_param_.train_path, "") << " train_path empty.";
    train_.reset(new mit::DMatrix(
      cli_param_.train_path, partid, npart, cli_param_.data_format));
    CHECK_NE(cli_param_.valid_path, "") << " valid_path empty.";
    valid_.reset(new mit::DMatrix(
      cli_param_.valid_path, partid, npart, cli_param_.data_format));
    
  } else if (cli_param_.task_type == "predict") {
    CHECK_NE(cli_param_.test_path, "") << " test_path empty.";
    test_.reset(new mit::DMatrix(
      cli_param_.test_path, partid, npart, cli_param_.data_format));
  }
  LOG(INFO) << "ps worker init done";
}

void Worker::Run() {
  CHECK_GT(cli_param_.batch_size, 0);
  std::vector<float> batch_metric;
  size_t progress_interval = cli_param_.batch_size * cli_param_.job_progress;
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
        if (progress % progress_interval == 0 && cli_param_.is_progress) {
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
    if (cli_param_.model == "mf"){
      SaveModel("", "user-");
    }
    static_cast<ps::SimpleApp *>(kv_worker_)->Request(mit::signal::METRIC, metric_info, ps::kScheduler);
  } // end for epochs
  if (cli_param_.model == "mf"){
    SaveModel("", "user-");
  } 
  // send signal to tell server & scheduler worker finish.
  kv_worker_->Wait(kv_worker_->Request(signal::WORKER_FINISH, "worker finish", ps::kScheduler + ps::kServerGroup));

  trainer_->timer_stats_->Print();
  LOG(INFO) << "@worker[" << ps::MyRank() << "] job finish.";
} 

void Worker::MiniBatch(const dmlc::RowBlock<mit_uint>& batch, std::vector<float>& train_metric) {
  /* pull request */
  // sorted unique key 
  trainer_->timer_stats_->begin(stats.ps_worker_pull);
  std::unordered_set<mit_uint> fset;

  std::unordered_map<mit_uint, int> fkv;
  bool extra = cli_param_.data_format == "libfm" && cli_param_.model == "ffm" ? true : false;
  std::vector<int> extras;
  
  //user key set for mf model
  std::unordered_set<mit_uint> user_set;
  //get fset. if model type is mf, get item_set, user_set, and rating_map
  if (cli_param_.model == "mf") {
    KeySetMF(batch, fset, user_set, rating_map);
  }
  else {
    KeySet(batch, fset, fkv, extra);
    if (extra) {
      extras.resize(keys.size(), 0);
      for (auto i = 0u; i < keys.size(); ++i) {
        if (fkv.find(keys[i]) == fkv.end()) continue;
        extras[i] = fkv[keys[i]];
      }
    }
  }
  // get keys (if the model is mf, get the item keys)
  std::vector<ps::Key> keys(fset.begin(), fset.end());
  sort(keys.begin(), keys.end());
  //pull operation (weight for the model. if model type is mf, then weights is item weight)
  std::vector<mit_float> weights;
  std::vector<int> lens;
  kv_worker_->Wait(kv_worker_->Pull(keys, extras, &weights, &lens));
  if (cli_param_.debug) {
    LOG(INFO) << "weights from server " << mit::DebugStr<mit_float>(weights.data(), 5);
  }
  trainer_->timer_stats_->stop(stats.ps_worker_pull);
  
  // worker computing
  // for mf model, grads store the item gradient(sgd) or result weight(als)
  std::vector<mit_float> grads(weights.size(), 0.0f);

  // for mf model, initialize the user weights
  if (cli_param_.model == "mf") {
    // for mf model, get the user keys
    std::vector<ps::Key> user_keys(user_set.begin(), user_set.end());
    sort(user_keys.begin(), user_keys.end());
    ps::KVPairs<mit_float> user_kv_pair;
    user_kv_pair.keys.CopyFrom(user_keys.data(), user_keys.size());
    model_->Pull(user_kv_pair, &user_weight_);
    if (cli_param_.debug) {
      LOG(INFO) << "keys from server: "
        << mit::DebugStr(keys.data(), keys.size());
      LOG(INFO) << "weights from server: "
        << mit::DebugStr(weights.data(), weights.size(), 12);
      LOG(INFO) << "lens from server: "
        << mit::DebugStr(lens.data(), lens.size());
      LOG(INFO) << "keys from worker: "
        << mit::DebugStr(user_kv_pair.keys.data(), user_kv_pair.keys.size());
      LOG(INFO) << "weights from worker: "
        << mit::DebugStr(user_kv_pair.vals.data(), user_kv_pair.vals.size(), 12);
      LOG(INFO) << "lens from worker: "
        << mit::DebugStr(user_kv_pair.lens.data(), user_kv_pair.lens.size());
    }
    //user gradient for mf(for mf model, user_grads store the user gradient(sgd) or result weight(als))
    std::vector<mit_float> user_grads(user_kv_pair.vals.size(), 0.0f);
    std::vector<mit_float> user_weights;
    std::vector<int> user_lens;
    user_weights.resize(user_kv_pair.vals.size());
    user_lens.resize(user_kv_pair.lens.size());
    for (size_t i = 0; i < user_kv_pair.lens.size(); ++i) {
      user_lens[i] = user_kv_pair.lens[i];
    }
    for (size_t i = 0; i < user_kv_pair.vals.size(); ++i) {
      user_weights[i] = user_kv_pair.vals[i];
    }
    if (cli_param_.debug) {
      LOG(INFO) << "user res vector before update:" << mit::DebugStr(user_grads.data(), user_grads.size(), 15);
      LOG(INFO) << "item res vector before update:" << mit::DebugStr(grads.data(), grads.size(), 12);
    }
    trainer_->Run(rating_map,
                  user_keys, user_weights, user_lens,
                  keys, weights, lens,
                  &user_grads, &grads);
    if (cli_param_.debug) {
      LOG(INFO) << "user res vector after update:" << mit::DebugStr(user_grads.data(), user_grads.size(), 15);
      LOG(INFO) << "item res vector after update:" << mit::DebugStr(grads.data(), grads.size(), 12);
    }
    //update user weights
    model_->Update(ps::SArray<mit_uint>(user_keys),
                   ps::SArray<mit_float> (user_grads),
                   ps::SArray<int>(user_lens),
                   &user_weight_);

  }
  else{  
    trainer_->Run(batch, keys, weights, lens, &grads, train_metric);
  }
  // push operation (gradient)
  kv_worker_->Push(keys, extras, grads, lens, mit::signal::UPDATE);
}

std::string Worker::Metric(mit::DMatrix* data) {
  std::vector<float> metrics(trainer_->MetricInfo().size(), 0.0f);
  std::vector<float> batch_metric(metrics.size(), 0.0f);
  auto metric_batch = cli_param_.batch_size * 10;
  auto metric_batch_count = 0l;
  std::string msg = "@w[" + std::to_string(ps::MyRank()) + "] valid <batch,inst> : <";
  data->BeforeFirst();
  while (true) {
    if (! data->Next()) break;
    auto& block = data->Value();
    uint32_t end = 0;
    for (auto i = 0u; i < block.size; i += metric_batch) {
      end = i + metric_batch > block.size ? block.size : i + metric_batch;
      const auto batch = block.Slice(i, end);
      MetricBatch(batch, batch_metric);
      for (auto idx = 0u; idx < batch_metric.size(); ++idx) {
        metrics[idx] += batch_metric[idx];
      }
      metric_batch_count += 1;
      if (cli_param_.is_progress) {
        LOG(INFO) << msg << metric_batch_count << "," << end << ">";
      }
    }
  } // while 

  for (auto& metric_value : metrics) metric_value /= metric_batch_count;
  
  return MetricMsg(metrics);
} // Worker::Metric

std::string Worker::MetricMsg(std::vector<float>& metrics) {
  std::string metric_info("");
  for (auto i = 0u; i < metrics.size(); ++i) {
    metric_info += const_cast<char *>(trainer_->MetricInfo()[i]->Name()) + std::string("^") + std::to_string(metrics[i]);
    if (i != metrics.size() - 1) metric_info += ",";
  }
  return metric_info;
} // MetricMsg


void Worker::MetricBatch(const dmlc::RowBlock<mit_uint>& batch, std::vector<float>& metrics_value) {
  std::unordered_set<mit_uint> fset;
  std::unordered_map<mit_uint, int> fkv;
  bool extra = cli_param_.data_format == "libfm" && cli_param_.model == "ffm" ? true : false;
  std::vector<int> extras;
  // user key set for mf model
  std::unordered_set<mit_uint> user_set;
  if (cli_param_.model == "mf") {
    //get keys (if the model is mf, get the item keys)
    KeySetMF(batch, fset, user_set, rating_map);
  }
  else {
    KeySet(batch, fset, fkv, extra);
    if (extra) {
      extras.resize(keys.size(), 0);
      for (auto i = 0u; i < keys.size(); ++i) {
        if (fkv.find(keys[i]) == fkv.end()) continue;
        extras[i] = fkv[keys[i]];
      }
    }
  }
  std::vector<ps::Key> keys(fset.begin(), fset.end());
  sort(keys.begin(), keys.end());
  
  // pull operation 
  std::vector<mit_float> weights;
  std::vector<int> lens; 
  kv_worker_->Wait(kv_worker_->Pull(keys, extras, &weights, &lens));

  // metric computing 
  metrics_value.clear();
  if (cli_param_.model == "mf") {
    // for mf model, get the user keys
    std::vector<ps::Key> user_keys(user_set.begin(), user_set.end());
    sort(user_keys.begin(), user_keys.end());
    // for mf model, pull the user weights
    ps::KVPairs<mit_float> user_kv_pair;
    user_kv_pair.keys.CopyFrom(user_keys.data(), user_keys.size());
    model_->Pull(user_kv_pair, &user_weight_);

    std::vector<mit_float> user_weights;
    std::vector<int> user_lens;
    user_weights.resize(user_kv_pair.vals.size());
    user_lens.resize(user_kv_pair.lens.size());
    for (size_t i = 0; i < user_kv_pair.lens.size(); ++i) {
      user_lens[i] = user_kv_pair.lens[i];
    }
    for (size_t i = 0; i < user_kv_pair.vals.size(); ++i) {
      user_weights[i] = user_kv_pair.vals[i];
    }
    trainer_->Metric(rating_map,
                     user_keys, user_weights, user_lens,
                     keys, weights, lens,
                     metrics_value);
  }
  else {
    trainer_->Metric(batch, keys, weights, lens, metrics_value);
  }
} // Worker::MetricBatch

void Worker::KeySet(const dmlc::RowBlock<mit_uint>& batch, 
                    std::unordered_set<mit_uint>& fset, 
                    std::unordered_map<mit_uint, int>& fkv, 
                    bool extra) {
  fset.clear();
  fset.insert(batch.index + batch.offset[0], batch.index + batch.offset[batch.size]); 
  fset.insert(0);
  if (extra) {
    #pragma omp parallel for num_threads(cli_param_.num_thread)
    for (auto i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
      #pragma omp critical
      fkv[batch.index[i]] = (int)batch.field[i];
    }
  }
} // method Worker::KeySet

void Worker::KeySetMF(const dmlc::RowBlock<mit_uint> & batch, 
                      std::unordered_set<mit_uint> & fset,
                      std::unordered_set<mit_uint> & user_set,
                      std::unordered_map<ps::Key, mit::mit_float> & rating_map) {
  CHECK_EQ(cli_param_.model, "mf");
  CHECK_EQ(cli_param_.data_format, "libsvm");
  for (size_t row_id = 0; row_id < batch.size; row_id++) {
    mit_uint user_id = (mit_uint)batch.label[row_id];
    user_set.insert(user_id);
    size_t length = batch.offset[row_id + 1] - batch.offset[row_id];
    for (size_t offset_index = 0; offset_index < length; offset_index++)
    {
      mit_uint item_id = batch.index[batch.offset[row_id] + offset_index];
      mit_float rating = batch.value[batch.offset[row_id] + offset_index];
      fset.insert(item_id);
      mit_uint new_key = mit::NewKey(
        user_id, item_id, cli_param_.nbit);
      if (rating_map.find(new_key) == rating_map.end()) {
        rating_map.insert(std::make_pair(new_key, rating));
      }
      if (cli_param_.debug) {
        //LOG(INFO) << "(" << user_id << "," << item_id << "," << rating << ")";
        //LOG(INFO) << user_id << " " << item_id << " " << new_key << " " << DecodeFeature(new_key, cli_param_.nbit)<<" "<< DecodeField(new_key, cli_param_.nbit) << " " << rating_map[new_key];
      }
    }
  }
} // Worker::KeySetMF

void Worker::SaveModel(std::string epoch, std::string prefix) {
  std::string myrank = std::to_string(ps::MyRank());
  LOG(INFO) << "@worker[" + myrank + "] save model begin";
  std::string dump_out = cli_param_.model_dump;
  std::string bin_out = cli_param_.model_binary;
  if (epoch == "") {
    dump_out += ("/" + prefix + "part-" + myrank);
    bin_out += ("/last/" + prefix + "part-" + myrank);
  } else {   // save middle result by epoch
    std::string postfix = "/" + prefix + "iter-" + epoch + "/part-" + myrank;
    dump_out += ".middle" + postfix;
    bin_out += postfix;
  }
  std::unique_ptr<dmlc::Stream> dumpfo(
    dmlc::Stream::Create(dump_out.c_str(), "w"));
  SaveTextModel(dumpfo.get());
  std::unique_ptr<dmlc::Stream> binfo(
    dmlc::Stream::Create(bin_out.c_str(), "w"));
  SaveBinaryModel(binfo.get());
  LOG(INFO) << "@server[" + myrank + "] save model done.";
}

void Worker::SaveTextModel(dmlc::Stream * fo) {
  std::unique_ptr<Transaction> trans(new Transaction(1, "server", "dump_out"));
  mit::EntryMeta * entry_meta = model_->EntryMeta();
  dmlc::ostream oss(fo);
  for (auto & kv : user_weight_) {
    auto key = kv.first;
    oss << key << "\t" << kv.second->String(entry_meta) << "\n";
  }
  // force flush before fo destruct 
  oss.set_stream(nullptr);
  Transaction::End(trans.get());
}

void Worker::SaveBinaryModel(dmlc::Stream * fo) {
  std::unique_ptr<Transaction> trans(
    new Transaction(1, "server", "binary_out"));
  // save entry meta
  mit::EntryMeta * entry_meta = model_->EntryMeta();
  entry_meta->Save(fo);
  // save model 
  std::unordered_map<ps::Key, mit::Entry * >::iterator iter;
  iter = user_weight_.begin();
  while (iter != user_weight_.end()) {
    // TODO  key special process
    fo->Write((char *) &iter->first, sizeof(ps::Key));
    iter->second->Save(fo, entry_meta);
    iter++;
  }
  Transaction::End(trans.get());
} // method SaveBinaryModel

} // namespace mit
