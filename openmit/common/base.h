/*!
 *  Copyright (c) 2016 by Contributors
 *  \file base.h
 *  \brief define macros and const variables
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_BASE_H_
#define OPENMIT_COMMON_BASE_H_

#include "dmlc/base.h"

namespace mit {
/*!
 * \brief unsigned integer type used in openmit
 */
typedef uint64_t mit_uint;

/*! \brief type real used in openmit */
typedef float mit_float;

inline mit_uint NewKey(mit_uint featid, size_t fieldid, size_t nbit) {
  return (featid << nbit) + fieldid;
}

inline mit_uint DecodeField(mit_uint new_key, size_t nbit) {
  return new_key % ((mit_uint)1 << nbit);
}

inline mit_uint DecodeFeature(mit_uint new_key, size_t nbit) {
  return new_key << nbit;
}

} // namemit mit
#endif // OPENMIT_COMMON_BASE_H_
