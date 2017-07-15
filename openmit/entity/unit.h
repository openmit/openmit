/*!
 *  Copyright (c) 2017 by Contributors
 *  \file unit.h
 *  \brief parameter computational unit that used to extract
 *        model interface and optimization algorithm interface
 *  \author ZhouYong
 */
#ifndef OPENMIT_ENTITY_UNIT_H_
#define OPENMIT_ENTITY_UNIT_H_

#include <cmath>
#include <unordered_map>
#include <vector>

#include "openmit/common/base.h"
#include "openmit/tools/dstruct/sarray.h"
#include "dmlc/logging.h"

namespace mit {
/*!
 * \brief model parameter unit.
 *        model     field_num   k   length(vec_)
 *        linear    0           0   1 + 0
 *        fm        1           k_  1 + 1 * k_
 *        ffm       field_num_  k_  1 + field_num_ * k_
 *        ...
 */
class Unit {
  public:
    /*! \brief constructor */
    Unit(size_t size, mit_float value = 0.0) : size_(size) {
      vec_.resize(size_, value);
    }

    /*! \brief destructor */
    ~Unit() {}

    /*! \brief copy from an iterator */
    inline void CopyFrom(
        std::vector<mit_float>::iterator begin,
        std::vector<mit_float>::iterator end);

    /*! \brief copy from another SArray */
    inline void CopyFrom(const mit::SArray<mit_float> & other);
    /*! \brief copy from a c-array */
    inline void CopyFrom(const mit_float * data, size_t size);

    /*! \brief weight, [linear, cross] */
    inline mit_float * Data() { return vec_.data(); }

    /*! \brief set unit parameter. weight or gradient */
    inline void Set(size_t i, mit_float value) {
      CHECK(i < size_); vec_[i] = value;
    }

    /*! \brief get data */
    inline mit_float Get(size_t i) {
      CHECK(i < size_) << "i:" << i << ",size:" << size_;
      return vec_[i];
    }

    /*! \brief length of unit */
    inline size_t Size() const { return size_; }

    /*! \brief return unit string */
    inline std::string Str() const;

    /*! \brief whether all zero value */
    inline bool AllZero() const;

  private:
    /*! parameter unit data structure. [linear_, cross_]*/
    mit::SArray<mit_float> vec_;
    /*! \brief */
    size_t size_;
};

inline void Unit::
CopyFrom(std::vector<mit_float>::iterator begin,
         std::vector<mit_float>::iterator end) {
  CHECK_EQ(end-begin, size_);
  vec_.CopyFrom(begin, end);
}

inline void Unit::
CopyFrom(const mit::SArray<mit_float> & other) {
  vec_.CopyFrom(other);
}

inline void Unit::
CopyFrom(const mit_float * data, size_t size) {
  vec_.CopyFrom(data, size);
}

inline std::string Unit::Str() const {
  std::stringstream ss;
  for (auto i = 0u; i < size_; ++i) {
    ss << vec_[i];
    if (i < size_ - 1) ss << ",";
  }
  return ss.str();
}

inline bool Unit::AllZero() const {
  bool b = true;
  for (auto i = 0u; i < size_; ++i) {
    if (fabs(vec_[i] - 0.0) > 1e-8) {
      b = false; break;
    }
  }
  return b;
}

/*!
 * here fieldth: 1 begin, kth: 0 begin . for example: field_num_=10, k_=4
 * index(fieldth, kth) = (fieldth-1)*k_ + kth + 1
 * index(2, 3) = 1 + (2-1)*4 + 3 = 8
 * last node: index(10,3) = 1 + (10-1)*4 + 3 = 40
 */
// parameter (server) map type
typedef std::unordered_map<mit_uint, mit::Unit *> PMAPT;

} // namespace mit

#endif // OPENMIT_ENTITY_UNIT_H_
