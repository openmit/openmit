/*!
 *  Copyright 2016 by Contributors
 * \file type_conversion.h
 * \brief basic type conversion
 * \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_UTIL_TYPE_CONVERSION_H_
#define OPENMIT_TOOLS_UTIL_TYPE_CONVERSION_H_

#include <sstream>
#include <stdint.h>
#include <string>

namespace mit {

/*!
 * \brief convert string to real type.
 *        std::string str = "12345678";
 *        uint32_t value = StringToNum<uint32_t>(str);
 */
template <typename Type>
Type StringToNum(const std::string & str) {
  istringstream buffer(str);
  Type value; buffer >> value;
  return value;
}

/*!
 * \brief 
 */
template <typename Type>
std::string NumToString(const Type & value) {
  ostringstream buffer;
  buffer << value;
  return std::string(buffer.str());
}

} // namespace mit

#endif // OPENMIT_TOOLS_UTIL_TYPE_CONVERSION_H_
