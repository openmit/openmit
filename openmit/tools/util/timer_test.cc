#include <stdio.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include "timer.h"

static std::string KEY1 = "key1";
static std::string KEY2 = "key2";

void test1() {
  double sum = 0.0;
  for (int i = 0; i < 1e10; ++i) {
    sum += i / 1000 + i%1000 + i*0.00001;
  }
}

void test2() {
  double sum = 0.0;
  for (long i = 0; i < 3*1e10; ++i) {
    sum += i / 1000 + i%1000 + i*0.00001;
  }
}

class TimerStats {
public:
  TimerStats() {} 
  ~TimerStats() {}
  
  std::string DebugString()  {
    std::unordered_map<std::string, double>::iterator iter = timer_stats_.begin();
    for (; iter != timer_stats_.end(); ++iter) {
      printf("key: %s, total time: %f\n", iter->first.c_str(), iter->second);
    }
    std::string str = "";
    return str;
  }

  void begin(std::string& key) {
    mu_.lock();
    double time = mit::GetTime();
    timer_recent_[key] = time;
    mu_.unlock();
  }

  void stop(std::string& key) {
    printf("stop key: %s\n", key.c_str());
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

private:
  std::unordered_map<std::string, double> timer_stats_;
  std::unordered_map<std::string, double> timer_recent_;
  std::mutex mu_;
};

int main(int argc, char * argv[]) {
  double timer = mit::GetTime();
  unsigned long timestamp = mit::TimeStamp();
  printf("timer: %f, timestamp: %ld\n", timer, timestamp);
  TimerStats ts;
  for (int i = 0; i < 5; ++i) {
  ts.begin(KEY1);
  test1();
  ts.stop(KEY1);
  }

  for (int i = 0; i < 3; ++i) {
  ts.begin(KEY2);
  test2();
  ts.stop(KEY2);
  }

  std::string rt = ts.DebugString();
  printf("rt: %s\n", rt.c_str());
  return 0;
}
