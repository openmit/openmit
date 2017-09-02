/*!
 *  Copyright (c) 2016 by Contributors
 *  \file base.h
 *  \brief define macros
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_BASE_H_
#define OPENMIT_COMMON_BASE_H_

#include "dmlc/base.h"

namespace mit {
/*!
 * \brief unsigned integer type used in mit,
 *        used for feature index, field index and row index.
 */
typedef uint64_t mit_uint;
/*! \brief float used for weight and calculation values. */
typedef float mit_float;
} // namemit mit
#endif // OPENMIT_COMMON_BASE_H_
