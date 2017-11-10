#include "openmit/framework/ps/scheduler.h"

namespace mit {

Scheduler::Scheduler(const mit::KWArgs & kwargs) {
  Init(kwargs);
}

Scheduler::~Scheduler() {
  // TODO
}

void Scheduler::Init(const mit::KWArgs & kwargs) {
  scheduler_.reset(new ps::SimpleApp(0));
  // register request processing func
  using namespace std::placeholders;
  scheduler_->set_request_handle(
    std::bind(&Scheduler::Handle, this, _1, _2));
  scheduler_->set_response_handle(
    std::bind(&Scheduler::Handle, this, _1, _2));
  
  complete_worker_number_ = 0;
  complete_server_number_ = 0;
}

void Scheduler::Run() {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this] { return exit_ == true; });
}

void Scheduler::Handle(const ps::SimpleData & recved, 
                       ps::SimpleApp * app) {
  ps::Message msg;
  msg.meta.head = recved.head;
  if (recved.body.size() > 0)
    msg.meta.body = recved.body;
  msg.meta.timestamp = recved.timestamp;
  msg.meta.request = false;
  msg.meta.simple_app = true;
  msg.meta.customer_id = scheduler_->get_customer()->id();
  msg.meta.recver = recved.sender;
  msg.meta.sender = ps::Postoffice::Get()->van()->my_node().id;
  // msg ready, send it
  ps::Postoffice::Get()->van()->Send(msg);

  int cmd = recved.head;
  switch(cmd) {
    case signal::METRIC:
      {
        mutex_.lock();
        UpdateMetric(recved);
        mutex_.unlock();
      }
      break;
    case signal::WORKER_FINISH:
      {
        mutex_.lock();
        complete_worker_number_++;
        mutex_.unlock();
      }
      break;
    default:
      LOG(FATAL) << "can not recognize signal.";
  }
}

void Scheduler::ExitCondition() {
  if (complete_worker_number_ == ps::NumWorkers()) {
    mutex_.lock();
    exit_ = true;
    mutex_.unlock();
    cond_.notify_all();
  }
}

void Scheduler::UpdateMetric(const ps::SimpleData & recved) {
  std::vector<std::string> datatypesets;
  std::vector<std::string> metrictypesets;
  std::unordered_map<std::string, int> removedepl;
  bool is_out = true;
  
  std::vector<std::string> metric_items;
  mit::string::Split(recved.body, & metric_items, ';');
  CHECK(metric_items.size() > 1) 
    << "metric info format: 'epoch;train:auc^0.8,logloss^0.1;...'";
  int epoch = std::atoi(metric_items[0].c_str());
  for (auto i = 1u; i < metric_items.size(); ++i) {
    std::vector<std::string> dataitems;
    mit::string::Split(metric_items[i], & dataitems, ':');
    std::string data_type = dataitems[0];
    CHECK(dataitems.size() == 2) 
      << "format error. train:auc^0.8,logloss^0.1";
    if (removedepl.find(data_type) == removedepl.end()) {
      datatypesets.push_back(data_type);
      removedepl.insert(std::make_pair(data_type, 1));
    }
    std::vector<std::string> datainfos;
    mit::string::Split(dataitems[1], & datainfos, ',');
    for (auto j = 0u; j < datainfos.size(); ++j) {
      std::vector<std::string> kv;
      mit::string::Split(datainfos[j], & kv, '^');
      CHECK(kv.size() == 2) << "metric info format error. auc^0.8";
      std::string metric_type = kv[0];     // "auc"/"logloss"/...
      if (removedepl.find(metric_type) == removedepl.end()) {
        metrictypesets.push_back(metric_type);
        removedepl.insert(std::make_pair(metric_type, 1));
      }
      auto value = std::atof(kv[1].c_str()); // 0.8 
      // stats
      auto key = data_type + "-" + metric_type;
      epoch_metric_number_[key][epoch] += 1;
      metric_sum_[key][epoch] += value;

      if (is_out && epoch_metric_number_[key][epoch] != ps::NumWorkers()) {
        is_out = false;
      }
    }
  } // for 
  // message output
  if (is_out) {
    std::string message = "finished epoch-" + std::to_string(epoch) + "";
    for (auto i = 0u; i < datatypesets.size(); ++i) {
      message += "\t[";
      auto data_type = datatypesets[i];
      message += data_type + "] ";
      for (auto j = 0u; j < metrictypesets.size(); ++j) {
        auto key = data_type + "-" + metrictypesets[j];
        auto value = metric_sum_[key][epoch] / ps::NumWorkers();
        message += metrictypesets[j] + ":" + std::to_string(value);
        message += (j != metrictypesets.size() - 1 ? "," : " ");
      }
    }
    LOG(INFO) << message;
  }
}

}// namespace mit
