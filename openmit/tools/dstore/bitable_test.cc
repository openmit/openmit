#include "openmit/tools/dstore/bitable.h"
#include "openmit/tools/hash/murmur3.h"
#include "openmit/tools/io/read.h"
#include <stdio.h>

int main(int argc, char * argv[]) {
  const char * file = argv[1];
  mit::MMap mmap_;
  bool is_open = mmap_.load(file);
  if (! is_open) printf("mmap_.open failed!");
  printf("file size: %d\n", (int) mmap_.size());

  //mit::dstore::ModelTable<float> model;
  //model.assign(mmap_.data());
  printf("done");
  return 0;
}
