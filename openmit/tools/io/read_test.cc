#include "./read.h"
#include <cstdio>

int main(int argc, char * argv[]) {
  const char * file = "read_test.cc";

  printf("================ [test] class MMap ================\n");
  mit::MMap mmap_;
  bool is_open = mmap_.load(file);
  if (!is_open) printf("mmap_.open failed!");
  printf("file size: %d\n", (int) mmap_.size());
  printf("file content:\n%s", (char *) mmap_.data());
  
  return 0;
}
