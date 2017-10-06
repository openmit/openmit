#include <iostream>
#include "openmit/entity/entry.h"

int main(int argc, char ** argv) {
  typedef uint64_t mit_uint;
  typedef float mit_float;

  mit_uint featid = 100000;
  mit_uint fieldid = 3;
  size_t embedding_size = 4;
  mit_uint arr[] = {2,3,5,6,7};
  size_t count = sizeof(arr) / sizeof(mit_uint);
  std::vector<mit_uint> fields(arr, arr + count);

  mit::Entry entry(featid, embedding_size, fields, fieldid);

  std::cout << "length: " << entry.length << std::endl;

  mit_float * v = entry.GetV(3);
  for (auto i = 0u; i < embedding_size; ++i) {
    std::cout << fields[3] << "\t v[" << i << "] " << v[i] << std::endl; 
  }
  std::cout << "model v ..." << std::endl;
  for (size_t i = 0; i < embedding_size; ++i) {
    entry.SetV(3, i, v[i] + 0.01);
    //v[i] += 0.01;
  }
  mit_float * v1 = entry.GetV(3);
  for (auto i = 0u; i < embedding_size; ++i) {
    std::cout << fields[3] << "\t v1[" << i << "] " << v1[i] << std::endl; 
  }
  return 0;
}
