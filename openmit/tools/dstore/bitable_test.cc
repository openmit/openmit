#include "openmit/tools/dstore/bitable.h"
#include "openmit/tools/io/read.h"
#include <stdio.h>

int main(int argc, char * argv[]) {
  const char * file = argv[1];
  mit::MMap mmap_;
  bool is_open = mmap_.load(file);
  if (! is_open) printf("mmap_.open failed!");
  printf("file size: %d\n", (int) mmap_.size());

  printf("done");

  mit::dstore::QuadSearch<float> model_;
  model_.assign(mmap_.data());
  std::cout << "coefficient_: " << model_.coefficient() << std::endl;
  std::cout << "value_num_: " << model_.value_num() << std::endl;
  std::cout << "bucket_size_: " << model_.bucket_size() << std::endl;
  std::cout << "model.size_: " << model_.size() << std::endl;
  std::string str = "openmit";
  const float * value = model_.find(str.c_str(), str.size());
  if (value) {
    std::cout << "str: " << str << ",\tvalue[0]: " << *(value + 0) << std::endl;
  } else {
    std::cout << "str: " << str << " not in model." << std::endl;
  }
  return 0;
}
