#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include "murmur3.h"
using namespace mit::hash;

int main(int argc, char * argv[]) {
  const char * feature_key = "openmit";
  uint32_t size = strlen(feature_key);

  uint64_t h3, h4;
  typedef MMHash128 hasher128_;
  hasher128_(feature_key, size, &h3, &h4, 0);
  printf("[2] key: %s, size: %d, h3: %lu, h4: %lu\n", feature_key, size, h3, h4);

  typedef MMHash64 hasher64_;
  uint64_t h5;
  hasher64_(feature_key, size, &h5, 0);
  printf("[640] key: %s, size: %d, h5: %lu\n", feature_key, size, h5);

  uint32_t h6;
  MMHash32(feature_key, size, &h6, 0);
  printf("[320] key: %s, size: %d, h6: %u\n", feature_key, size, h6);

  return 0;
}
