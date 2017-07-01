 /*!
 *  Copyright (c) 2017 by Contributors
 *  \file bitable.h
 *  \brief memory-based binary big table structure.
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_DSTORE_BITABLE_H_
#define OPENMIT_TOOLS_DSTORE_BITABLE_H_

#include "openmit/tools/hash/murmur3.h"
#include <string>
#include <iostream>

namespace mit {
namespace dstore {
/*! 
 * \brief quadratic search model table structure
 */
template <typename VType, typename Hasher = mit::hash::MMHash128>
struct QuadSearch {
  /* type structure defination */
  // index. first_key & (num_buckets - 1)
  typedef uint32_t idx_type;
  typedef const idx_type * const_idx_pointer;
  // key. second_key. used to sequential search
  typedef uint64_t key_type;
  typedef const key_type * const_key_pointer;
  // value. data content stored
  typedef VType value_type;
  typedef const VType * const_value_pointer;
  // record. record data: key|value
  typedef unsigned char record_type;
  typedef const record_type * const_record_pointer;
  // bucket interval. [first, last] for each bucket range.
  typedef std::pair<idx_type, idx_type> bucket_type;
  typedef const bucket_type * const_bucket_pointer;
  // offset
  typedef uint32_t uint;
  
  /* initialize constants */
  static const uint idx_size = sizeof(idx_type);
  static const uint key_size = sizeof(key_type);
  static const uint value_size = sizeof(VType);
  static const uint record_size = key_size + value_size; // record: key|value

  /*! \brief load model data to bi-model */
  void assign(const void * data) {
    idx_ = const_idx_pointer(data);
    idx_type bucket_size = *idx_++;   // first element: num_buckets
    std::cout << "bucket_size; " << bucket_size << std::endl;
    bucket_1_ = bucket_size - 1;
    record_ = const_record_pointer(idx_ + bucket_size + 1);
    size_ = (2 + bucket_size) * idx_size + idx_[bucket_size] * record_size;
    
    const_record_pointer first = record_, last = record_ + 15 * record_size;
    for (; first != last; first += record_size) {
      uint64_t h2 = *const_key_pointer(first);
      float value = *const_value_pointer(first + key_size);
      printf("key: %lu, value: %0.8f\n", h2, value);
    }
    std::cout << "size_: " << size_ << std::endl;
    std::string str = "zhouyong";
    const_value_pointer zvalue = find(str.c_str(), str.size());
    if (zvalue == 0) {
      std::cout << "str: " << str << " not exists. \t zvalue: 0. " << std::endl;
    } else {
      std::cout << "str: " << str << " exists. \t zvalue: " << *zvalue << std::endl;
    }
  }

  /*! \brief find value according data and size */
  const_value_pointer find(const void * data, int size) const {
    uint64_t h1, h2;
    hasher_(data, size, &h1, &h2, 0);
    std::cout << "data: " << data << ", h1: " << h1 << ",\th2: " << h2 << std::endl;
    return search(h1, h2);
  }

  /*! \brief bi-model size */
  uint64_t size() const { return size_; }

  private:
    /*! \brief search record data according h1 and h2 */
    const_value_pointer search(uint64_t h1, uint64_t h2) const {
      // h1: find index
      uint64_t bucket_index = (h1 & bucket_1_);
      std::cout << "bucket_index: " << bucket_index << std::endl;
      const_bucket_pointer bucket = 
        const_bucket_pointer(idx_ + bucket_index);

      const_record_pointer first = record_ + bucket->first * record_size;
      const_record_pointer last = record_ + bucket->second * record_size;
      // h2: sequential search
      for (; first != last; first += record_size) {
        if (*const_key_pointer(first) != h2) continue;
        return const_value_pointer(first + key_size);
      }
      return 0;
    }

  private:
    /*! \brief hash mathod for generate index and key */
    typedef Hasher hasher_;
    /*! \brief index of buckets upper boundary */
    idx_type bucket_1_;
    /*! \brief index data region */
    const_idx_pointer idx_;
    /*! \brief record data region */
    const_record_pointer record_;
    /*! \brief offset */
    uint64_t size_;

}; // struct QuadSearch

/*! \brief factorization model value structure. default k=4 */
struct FMValue {
  float weight;
  float vec[4];
};

} // namespace dstore

} // namespace mit

#endif // OPENMIT_TOOLS_DSTORE_BITABLE_H_
