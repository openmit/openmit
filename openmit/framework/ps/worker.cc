#include "openmit/framework/ps/worker.h"
#include "openmit/tools/monitor/transaction.h"

namespace mit {

Worker::Worker(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

Worker::~Worker() {
  delete kv_worker_;
}

void Worker::Init(const mit::KWArgs & kwargs) {
  std::unique_ptr<mit::Transaction> trans(
    mit::Transaction::Create(2, "worker", "init"));
  // 1. param_
  cli_param_.InitAllowUnknown(kwargs);
  // 2. kv_worker_
  kv_worker_ = new ps::KVWorker<mit_float>(0);
  // 3. trainer_
  trainer_.reset(new mit::Trainer(kwargs));
  // 5. data
  int partid = ps::MyRank();
  int npart = ps::NumWorkers();
  LOG(INFO) << "partid: " << partid << ", npart: " << npart;
  if (cli_param_.task_type == "train") {
    CHECK_NE(cli_param_.train_path, "") 
      << " train_path is empty! need train_path.";
    train_.reset(new mit::DMatrix(
          cli_param_.train_path, partid, npart, cli_param_.data_format));
    CHECK_NE(cli_param_.valid_path, "") 
      << " valid_path is empty! need evalution_path.";
    valid_set_.reset(new mit::DMatrix(
          cli_param_.valid_path, partid, npart, cli_param_.data_format));
    
    InitFSet(train_.get(), & train_fset_);
    InitFSet(valid_set_.get(), & valid_fset_);
    //std::unordered_set<ps::Key> fset;
    //fset.insert(0);
    //valid_set_->BeforeFirst();
    //while (valid_set_->Next()) {
    //  const auto & block = valid_set_->Value();
    //  fset.insert(
    //    block.index + block.offset[0], 
    //    block.index + block.offset[block.size]);
    //}
    //.insert(valid_fset_.end(), fset.begin(), fset.end());
    //sort(.begin(), valid_fset_.end());
  } else if (cli_param_.task_type == "predict") {
    CHECK_NE(cli_param_.test_path, "")
      << " test_path is empty! need test_path.";
    test_set_.reset(new mit::DMatrix(
          cli_param_.test_path, partid, npart, cli_param_.data_format));
  }
  mit::Transaction::End(trans.get());
}

void Worker::InitFSet(mit::DMatrix * data, std::vector<ps::Key> * feat_set) {
  std::unordered_set<ps::Key> fset;
  fset.insert(0);
  data->BeforeFirst();
  while (data->Next()) {
    const auto & block = data->Value();
    fset.insert(
        block.index + block.offset[0],
        block.index + block.offset[block.size]);
  }
  feat_set->insert(feat_set->end(), fset.begin(), fset.end());
  sort(feat_set->begin(), feat_set->end());
}

void Worker::Run() {
  std::unique_ptr<mit::Transaction> trans(
    mit::Transaction::Create(1, "ps", "worker"));
  CHECK_GT(cli_param_.batch_size, 0) << "[ERROR] batch_size <= 0."; 
  size_t progress_interval = cli_param_.batch_size * cli_param_.job_progress;
  for (auto epoch = 0u; epoch < cli_param_.max_epoch; ++epoch) {
    uint64_t progress = 0u;
    train_->BeforeFirst();
    while (train_->Next()) { 
      auto & block = train_->Value();
      uint32_t end = 0;
      for (auto i = 0u; i < block.size; i += cli_param_.batch_size) {
        end = i + cli_param_.batch_size >= block.size ? 
          block.size : i + cli_param_.batch_size;
        if (progress % progress_interval == 0 && cli_param_.is_progress) {
          LOG(INFO) << "@worker[" << ps::MyRank() << "] progress \
                    <epoch, inst>: <" << epoch << ", "<< progress << ">";
        }
        progress += (end - i);
        if ((end - i) != cli_param_.batch_size && cli_param_.is_progress) {
          LOG(INFO) << "@worker[" << ps::MyRank() << "] progress \
                    <epoch, inst>: <" << epoch << ", "<< progress << ">";
        }
        const auto batch = block.Slice(i, end);
        MiniBatch(batch);
      }
    } 
    // TODO Barrier
    if (cli_param_.save_peroid != 0 && epoch % cli_param_.save_peroid == 0) {
      kv_worker_->Push({epoch}, {}, {}, signal::SAVEINFO);
    }
    
    // evaluation based on lastest model
    float metric_train = Metric(train_.get(), train_fset_);
    float metric_eval = Metric(valid_set_.get(), valid_fset_);
    
    std::string metric_info = 
      cli_param_.metric + "," + std::to_string(epoch) + "," + std::to_string(metric_train) 
      + "," + cli_param_.metric + "," + std::to_string(epoch) + "," + std::to_string(metric_eval);
    
    static_cast<ps::SimpleApp *>(kv_worker_)->Request(
        mit::signal::METRIC, metric_info, ps::kScheduler);
  } // end for epoch
  
  kv_worker_->Wait(kv_worker_->Request(
        signal::WORKER_COMPLETE, "worker", ps::kScheduler));

  //kv_worker_->Request(signal::WORKER_COMPLETE, "worker", ps::kScheduler);
  
  // message to tell server job finish
  /*
  std::vector<ps::Key> keys(1,1);
  std::vector<float> vals(1,1);
  std::vector<int> lens(1,1);
  kv_worker_->Wait(
      kv_worker_->Push(keys, vals, lens, mit::signal::FINISH));
  */
  mit::Transaction::End(trans.get());
  LOG(INFO) << "@worker[" << ps::MyRank() << "] epoch completation";
} 

float Worker::Metric(mit::DMatrix * data, std::vector<ps::Key> & feat_set) {
  // pull (weight)
  std::vector<mit_float> rets;
  kv_worker_->Wait(kv_worker_->Pull(feat_set, & rets));
  // worker metric
  float metric = trainer_->Eval(data, feat_set, rets);
  return metric;
}

void Worker::MiniBatch(const dmlc::RowBlock<mit_uint> & batch) {
  // step3: Pull(newID, &vals, nullptr, cmd=DecodeFeature). cmd if row.field != null
  // step4: Pull Retry Number Control
  // step5: worker computing 
  // step1: if row.field != null: generate newID according to (featureid, fieldid)
  std::unordered_set<mit_uint> fset;
  if (cli_param_.data_format == "libfm") {
    for (auto i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
      mit_uint new_key = batch.index[i];
      if (new_key > 0) {
        new_key = mit::NewKey(batch.index[i], batch.field[i], cli_param_.nbit);
      }
      fset.insert(new_key);   
    }
  } else { // data_format in ["auto", "libsvm"]
    fset.insert(batch.index + batch.offset[0], batch.index + batch.offset[batch.size]);
  }
  fset.insert(0);  // for bias
  std::vector<ps::Key> keys(fset.begin(), fset.end());
  sort(keys.begin(), keys.end());
  
  // pull operation (weight)
  std::vector<mit_float> weights;
  std::vector<int> lens; 
  kv_worker_->Wait(
    kv_worker_->Pull(keys, &weights, &lens));

  if (cli_param_.debug) {
    LOG(INFO) << "weights from server: " 
      << mit::DebugStr<mit_float>(weights.data(), 10, 10);
  }
  
  // worker computing 
  std::vector<mit_float> grads(weights.size(), 0.0f);
  trainer_->Run(batch, keys, weights, lens, &grads);

  // push operation (gradient)
  kv_worker_->Wait(
    kv_worker_->Push(keys, grads, lens, mit::signal::UPDATE));
}

} // namespace mit
