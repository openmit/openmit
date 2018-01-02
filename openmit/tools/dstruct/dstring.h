/*!
 *  Copyright (c) 2016 by Contributors
 *  \file dstring.h
 *  \brief string basic op 
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_DSTRUCT_DSTRING_H_
#define OPENMIT_TOOLS_DSTRUCT_DSTRING_H_

#include <string>
#include <sstream>
#include <vector>

namespace mit {
namespace string {

inline void Split(const std::string & str, 
                  std::vector<std::string> * ret,
                  const char delimeter = ' ') {
  std::string item;
  std::istringstream is(str);
  ret->clear();
  while(std::getline(is, item, delimeter)) {
    ret->push_back(item);
  }
}

} // namespace string
} // namespace mit

#endif // OPENMIT_TOOLS_DSTRUCT_DSTRING_H_
