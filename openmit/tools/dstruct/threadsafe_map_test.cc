#include "threadsafe_map.h"

int main(int argc, char** argv) {
  mit::ThreadsafeMap<int, int> map;
  int M = 20;
  for (auto i = 0; i < M; ++i) {
    int x = i * 100;
    int* v = &x;
    map.insert(i, v);
    auto* result = map[i];
    printf("i: %d, result: %d\n", i, *result);
  }

  for (auto i = 0; i < M; ++i) {
    auto* result = map[i];
    //printf("i: %d, v: %d\n", i, *map[i]);
  }
  return 0;
}
