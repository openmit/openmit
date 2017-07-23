/*!
 *  Copyright (c) 2017 by Contributors
 *  \file ffm_table.h
 *  \brief memory-based binary ffm model structure.
 *  \author ZhouYong
 */
#include <iostream>
#include <unordered_map>
#include <vector>
#include "openmit/tools/dstore/ffm_table.h"

int main(int argc, char ** agrv) {
  uint32_t embedding_size = 4;
  mit::dstore::FFMEntry node(embedding_size);
  uint32_t size = 1 + (1 + embedding_size) * 3;
  std::vector<float> v(size, 0.0);
  v[0] = 0.0001;
  for (auto i = 1u; i < size; i += (1 + embedding_size)) {
    v[i] = i;
    for (auto j = 1u; j <= embedding_size; ++j) {
      v[i+j] = (i+j) * 0.1;
    }
  }
  node.assign(std::make_pair(size, v.data()));
  float * vif = node.Get(6);
  if (vif != nullptr) {
    for (auto i = 0u; i < embedding_size; ++i) {
      std::cout << vif[i] << " ";
    }
    std::cout << std::endl;
  } else {
    std::cout << "vif not exists." << std::endl;
  }
  return 0;
}
