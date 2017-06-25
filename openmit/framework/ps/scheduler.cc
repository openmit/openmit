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
  scheduler_->set_request_handle(std::bind(&Scheduler::Handle, 
        this, std::placeholders::_1, std::placeholders::_2));
  scheduler_->set_response_handle(std::bind(&Scheduler::Handle, 
        this, std::placeholders::_1, std::placeholders::_2));
  complete_worker_number_ = 0;
  complete_server_number_ = 0;
}

void Scheduler::Run() {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this] { return exit_ == true; });
}

void Scheduler::Handle(const ps::SimpleData & reqinfo, ps::SimpleApp * app) {
  ps::Message msg;
  msg.meta.head = reqinfo.head;
  if (reqinfo.body.size() > 0)
    msg.meta.body = reqinfo.body;
  msg.meta.timestamp = reqinfo.timestamp;
  msg.meta.request = false;
  msg.meta.simple_app = true;
  msg.meta.customer_id = scheduler_->get_customer()->id();
  msg.meta.recver = reqinfo.sender;
  msg.meta.sender = ps::Postoffice::Get()->van()->my_node().id;
  ps::Postoffice::Get()->van()->Send(msg);

  int cmd = reqinfo.head;
  switch(cmd) {
    case signal::METRIC:
      {
        mutex_.lock();
        std::cout << "scheduler METRIC msg.meta.customer_id: " <<msg.meta.customer_id << std::endl;
        UpdateMetric(reqinfo);
        mutex_.unlock();
      }
      break;
    case signal::WORKER_COMPLETE:
      {
        mutex_.lock();
        std::cout << "scheduler WORKER_COMPLETE msg.meta.customer_id: " <<msg.meta.customer_id << std::endl;
        complete_worker_number_++;
        mutex_.unlock();
      }
      break;
    default:
      LOG(ERROR) << "can not recognize signal.";
  }
}

void Scheduler::ExitCondition() {
  std::cout << "complete_worker_number_: " << complete_worker_number_ << ", ps::NumWorkers(): " << ps::NumWorkers() 
    << "complete_server_number_: " << complete_server_number_ << ", ps::NumServers(): " << ps::NumServers() << std::endl;
  if (complete_worker_number_ == ps::NumWorkers()) {
    std::cout << "Scheduler::ExitCondition begin" << std::endl;
    mutex_.lock();
    exit_ = true;
    mutex_.unlock();
    std::cout << "Scheduler::ExitCondition begin notify_all" << std::endl;
    cond_.notify_all();
    std::cout << "Scheduler::ExitCondition done notify_all" << std::endl;
    std::cout << "Scheduler::ExitCondition done" << std::endl;
  }
}

void Scheduler::MetricInfo(const std::string & data_type,
                           const std::string & metric_type,
                           int epoch,
                           float metric_value) {
  if (data_type == "train") {
    epoch_metric_number_train_[metric_type][epoch] += 1;
    metric_sum_train_[metric_type][epoch] += metric_value;
  } else if (data_type == "eval") {
    epoch_metric_number_eval_[metric_type][epoch] += 1;
    metric_sum_eval_[metric_type][epoch] += metric_value;
  }

  if (epoch_metric_number_train_[metric_type][epoch] == ps::NumWorkers() && 
      epoch_metric_number_eval_[metric_type][epoch] == ps::NumWorkers()) {
    LOG(INFO) << "[" << epoch << "]\ttrain[" << metric_type << "]: " 
      << metric_sum_train_[metric_type][epoch] / ps::NumWorkers() 
      << "\ttest[" << metric_type << "]: " 
      << metric_sum_eval_[metric_type][epoch] / ps::NumWorkers();
  }
}

void Scheduler::UpdateMetric(const ps::SimpleData & recved) {
  std::vector<std::string> metric;
  mit::Split(recved.body, ',', & metric);
  CHECK_EQ(metric.size(), 6);
  MetricInfo("train", 
      metric[0], std::atoi(metric[1].c_str()), std::atof(metric[2].c_str()));
  MetricInfo("eval", 
      metric[3], std::atoi(metric[4].c_str()), std::atof(metric[5].c_str()));
}

} // namespace mit
