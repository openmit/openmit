#include <time.h>
#include "thread_pool.h"

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

int main(int argc, char** argv) {
  int M = 16;
  long N = 1000000000;
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

  
  return 0;
}
