/*!
 *  Copyright (c) 2016 by Contributors
 *  \file string_op.h
 *  \brief basic match formula, such as sigmoid, distance etc.
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_DSTRUCT_STRING_OP_H_
#define OPENMIT_TOOLS_DSTRUCT_STRING_OP_H_

#include <string>
#include <sstream>
#include <vector>

namespace mit {

inline void Split(const std::string & str, 
                  char delimeter, 
                  std::vector<std::string> * ret) {
  std::string item;
  std::istringstream is(str);
  ret->clear();
  while(std::getline(is, item, delimeter)) {
    ret->push_back(item);
  }
}

} // namespace mit

#endif // OPENMIT_TOOLS_DSTRUCT_STRING_OP_H_
