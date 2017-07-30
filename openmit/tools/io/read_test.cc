#include "./read.h"
#include <cstdio>

int main(int argc, char * argv[]) {
  const char * file = "io_example.txt";

  printf("================ [test] class MMap ================\n");
  mit::MMap mmap_;
  bool is_open = mmap_.load(file);
  if (!is_open) printf("mmap_.open failed!");
  printf("file size: %d\n", (int) mmap_.size());
  printf("file content:\n%s", (char *) mmap_.data());
 
  printf("================ [test] class Read ================\n");
  mit::Read read(file, false);
  std::string line;
  while (read.get_line(line)) {
    printf("line: %s\n", line.c_str());
  }

  return 0;
}
