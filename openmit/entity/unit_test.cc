/*!
 * Copyright (c) 2017 by Contributors
 * \file unit_test.cc
 * \brief computational unit test
 * \author ZhouYong
 */
#include "openmit/entity/unit.h"
#include <iostream>

int main(int argc, char * argv[]) {
  size_t size = 10;
  mit::Unit * unit = new mit::Unit(size, 0.2f);
  std::cout << "size: " << unit->Size() << std::endl;
  std::cout << "element: ";
  for (auto i = 0u; i < size; ++i) {
    std::cout << unit->Get(i) << " ";
  }
  std::cout << std::endl;
  std::cout << "allZero: " << unit->AllZero() << std::endl;
  return 0;
}
