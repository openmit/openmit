#ifndef OPENMIT_TOOLS_PROFILER_TIMER_STATS_H_
#define OPENMIT_TOOLS_PROFILER_TIMER_STATS_H_

#include <mutex>
#include <string> 
#include <unordered_map>
#include "dmlc/logging.h"
#include "openmit/tools/util/timer.h"

namespace mit {

struct STATS {
  std::string ps_worker = "worker";
	std::string ps_worker_io = "worker-io";
  std::string ps_worker_train = "worker-train";
	std::string ps_worker_train_metric = "worker-train_metric";
	std::string ps_worker_train_eval = "worker-train_eval";
	std::string ps_worker_train_send_metric = "worker-train_send_metric";
  std::string ps_worker_pull = "worker-pull";
  std::string ps_worker_calc_grad = "worker-calc_grad";
  std::string ps_worker_calc_loss = "worker-loss_calc";
  std::string ps_worker_model_predict = "worker-model_predict";
  std::string ps_worker_model_gradient = "worker-model_gradient";
  std::string ps_worker_valid = "worker-valid";
	std::string ps_server_pull_request = "server-pull_request";
  std::string ps_server_update = "server-update";
};

struct StatsUnit {
  int           level;
  std::string   key;
  double        time_consuming;
}; // struct StatsUnit

class TimerStats {
public:
  TimerStats() {} 
  ~TimerStats() {}

  void begin(std::string& key) {
    mu_.lock();
    double time = mit::GetTime();
    timer_recent_[key] = time;
    mu_.unlock();
  }

  void stop(std::string& key) {
    if (timer_recent_.find(key) == timer_recent_.end()) {
      printf("error to not exist key %s\n", key.c_str());
      return;
    }
    double sum = 0.0;
    if (timer_stats_.find(key) != timer_stats_.end())
      sum = timer_stats_[key];
    sum += (mit::GetTime() - timer_recent_[key]);
    mu_.lock();
    timer_stats_[key] = sum;
    mu_.unlock();
  }
  
  std::string DebugString()  {
    std::unordered_map<std::string, double>::iterator iter = timer_stats_.begin();
    for (; iter != timer_stats_.end(); ++iter) {
      printf("key: %s, total time: %f\n", iter->first.c_str(), iter->second);
    }
    std::string str = "";
    return str;
  }

  void Print() {
    double total = 0.0;
    std::unordered_map<std::string, double>::iterator iter = timer_stats_.begin();
    for (; iter != timer_stats_.end(); iter++) {
      if (iter->first == "worker-train" || iter->first == "worker-train_eval") continue;
      total += iter->second;
    }
    LOG(INFO) << "\n================ print timer stats info ====================\n";
    LOG(INFO) << "key\t\ttime(s)\t\tpercent";
    iter = timer_stats_.begin();
    while (iter != timer_stats_.end()) {
      std::string record(iter->first);
      record += "\t" + std::to_string(iter->second) + "\t" + std::to_string(iter->second * 100 / total) + "%";
      LOG(INFO) << record;
      iter++;
    }
  }

private:
  std::unordered_map<std::string, double> timer_stats_;
  std::unordered_map<std::string, double> timer_recent_;
  std::mutex mu_;
  /*! \brief store each time-consuming result */
  std::unordered_map<std::string, TimerUnit> time_consuming_;
}; // class TimerStats

} // namespace mit 
#endif // OPENMIT_TOOLS_PROFILER_TIMER_STATS_H_ 
