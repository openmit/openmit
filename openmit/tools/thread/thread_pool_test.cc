#include <time.h>
#include "thread_pool.h"
#include "openmit/tools/dstruct/threadsafe_map.h"
#include <unordered_map>
#include <unistd.h>

double GetTime(void) {
  timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
    return static_cast<double>(ts.tv_sec) + 
      static_cast<double>(ts.tv_nsec) * 1e-9;
  } else {
    return static_cast<double>(time(NULL));
  }
}
long for_sum(long N) {
  long sum = 0l;
  #pragma omp parallel for reduction(+:sum) num_threads(4)
  for (long i = 0; i < N; ++i) {
    sum += i;
  }
  return sum;
}
void Insert(mit::ThreadsafeMap<int, int>& map, long N) {
  for (long i = 0; i < N; ++i) {
    map.insert((int)i, int(i));
  }
}
/*
void Insert(std::unordered_map<int, int>& map, long N) {
  for (long i = 0; i < N; ++i) {
    map.insert(std::make_pair((int)i, (int)i));
  }
}
*/

int main(int argc, char** argv) {
  int M = 8;
  long N = 100;
  double start_time = GetTime();
  long sum = 0l;
  for (int i = 0; i < M; ++i) {
    sum += for_sum(N*i);
  }
  printf("single thread: %f, sum: %ld\n", (GetTime() - start_time), sum);
  
  start_time = GetTime();
  mit::ThreadPool thread_pool(M);
  std::vector<std::future<long>> vec;
  for (int i = 0; i < M; ++i) {
    vec.emplace_back(thread_pool.Append([i, N] { 
      long result = for_sum(N*i); 
      return result; 
    }
    ));
  }
  sum = 0l;
  for (int i = 0; i < M; ++i) sum += vec[i].get();
  printf("thread pool: %f, sum: %ld\n", (GetTime() - start_time), sum);

  /*
  std::unordered_map<int, int> map;
  map.insert(std::make_pair(1,1));
  if (map.find(10) == map.end()) {
    printf("1. 10 not in map\n");
  } else {
    printf("1. 10 in map\n");
  }
  if (map.find(10) == map.end()) {
    printf("2. 10 not in map\n");
  } else {
    printf("2. 10 in map\n");
  }
  */

  mit::ThreadsafeMap<int, int> map;
  //std::unordered_map<int, int> map;
  mit::ThreadPool pool(M/2);
  for (int i = 0; i < M; ++i) {
    pool.Append([i, &map, N] { Insert(map, N); });
  }
  printf("\n================================\n");
  for (long i = 0; i < N; ++i) { 
    if (map.find(i) == map.end()) {
      printf("1-key: %ld not in map\n", i);
    } else {
      printf("1-key: %ld, value: %d\n", i, map[i]);
    }
  }
  printf("sleep 1s\n");
  sleep(1);
  printf("\n================================\n");
  for (long i = 0; i < N; ++i) { 
    printf("2-key: %ld, value: %d\n", i, map[i]);
  }
  
  return 0;
}
