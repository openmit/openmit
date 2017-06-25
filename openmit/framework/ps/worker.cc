#include "openmit/framework/ps/worker.h"

namespace mit {

DMLC_REGISTER_PARAMETER(WorkerParam);

Worker::Worker(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

Worker::~Worker() {
  delete kv_worker_;
}

void Worker::Init(const mit::KWArgs & kwargs) {
  // 1. param_
  param_.InitAllowUnknown(kwargs);
  // 2. kv_worker_
  kv_worker_ = new ps::KVWorker<mit_float>(0);
  // 3. trainer_
  trainer_.reset(new mit::Trainer(kwargs));
  // 5. data
  int partid = ps::MyRank();
  int npart = ps::NumWorkers();
  LOG(INFO) << "partid: " << partid << ", npart: " << npart;
  if (param_.task == "train") {
    CHECK_NE(param_.train_path, "") 
      << " train_path is empty! need train_path.";
    train_set_.reset(new mit::DMatrix(
          param_.train_path, partid, npart, param_.data_format));
    CHECK_NE(param_.valid_path, "") 
      << " valid_path is empty! need evalution_path.";
    valid_set_.reset(new mit::DMatrix(
          param_.valid_path, partid, npart, param_.data_format));
    
    InitFSet(train_set_.get(), & train_fset_);
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
  }
  if (param_.task == "predict") {
    CHECK_NE(param_.test_path, "")
      << " test_path is empty! need test_path.";
    test_set_.reset(new mit::DMatrix(
          param_.test_path, partid, npart, param_.data_format));
  }
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
  CHECK_GT(param_.batch_size, 0) << " error: batch_size <= 0.";
  
  for (auto iter = 0u; iter < param_.max_epoch; ++iter) {
    // training based on epoch
    train_set_->BeforeFirst();
    while (train_set_->Next()) { 
      const dmlc::RowBlock<mit_uint> & block = train_set_->Value();
      int end = 0;
      for (auto i = 0u; i < block.size; i += param_.batch_size) {
        if (i + param_.batch_size <= block.size) {
          end = i + param_.batch_size;
        } else {
          end = block.size;
        }
        const auto batch = block.Slice(i, end);
        MiniBatch(batch);
      }
    } 
    // TODO Barrier
    if (param_.save_peroid != 0 && iter % param_.save_peroid == 0) {
      kv_worker_->Push({iter}, {}, {}, signal::SAVEINFO);
    }
    
    // evaluation based on lastest model
    float metric_train = Metric(train_set_.get(), train_fset_);
    float metric_eval = Metric(valid_set_.get(), valid_fset_);
    
    std::string metric_info = 
      param_.metric + "," + std::to_string(iter) + "," + std::to_string(metric_train) 
      + "," + param_.metric + "," + std::to_string(iter) + "," + std::to_string(metric_eval);
    
    static_cast<ps::SimpleApp *>(kv_worker_)->Request(
        mit::signal::METRIC, metric_info, ps::kScheduler);
  } // end for iter
  
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
  std::unordered_set<mit_uint> fset(
      batch.index + batch.offset[0], batch.index + batch.offset[batch.size]);
  fset.insert(0);  // for bias
  std::vector<ps::Key> keys(fset.begin(), fset.end());
  sort(keys.begin(), keys.end());
  
  // pull operation (weight)
  std::vector<int> lens(keys.size(), param_.field_num * param_.k + 1);
  std::vector<mit_float> rets;
  kv_worker_->Wait(kv_worker_->Pull(keys, &rets));
  
  // worker computing 
  std::vector<mit_float> vals;
  trainer_->Run(batch, keys, rets, &vals);

  // push operation (gradient)
  kv_worker_->Wait(
      kv_worker_->Push(keys, vals, lens, mit::signal::UPDATE));
}

} // namespace mit
