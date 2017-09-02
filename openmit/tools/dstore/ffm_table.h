/*!
 *  Copyright (c) 2017 by Contributors
 *  \file ffm_table.h
 *  \brief memory-based binary ffm model structure.
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_DSTORE_FFM_TABLE_H_
#define OPENMIT_TOOLS_DSTORE_FFM_TABLE_H_

#include <unordered_map>
#include <vector>
#include "openmit/tools/hash/murmur3.h"

namespace mit {
namespace dstore {
/*!
 * \brief ffm model structure
 */
template <typename VType, typename Hasher = mit::hash::MMHash128>
struct FFMTable {
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
  // count
  typedef uint32_t uint;
  // model ffm entry
  typedef std::pair<uint32_t, const float *> entry_type;
  
  /* initialize constants */
  static const uint idx_size = sizeof(idx_type);
  static const uint key_size = sizeof(key_type);
  static const uint count_size = sizeof(uint);
  static const uint value_size = sizeof(VType);
  
  /*! \brief load model data to bi-model */
  void assign(const void * data) {
    idx_ = const_idx_pointer(data);
    bias_coefficient_ = *idx_++;      // first : bias coefficient
    embedding_size_ = *idx_++;        // second: embedding_size
    idx_type bucket_size = *idx_++;   // third: num_buckets
    bucket_1_ = bucket_size - 1;
    record_ = const_record_pointer(idx_ + bucket_size + 1);
    size_ = (4 + bucket_size) * idx_size + idx_[bucket_size];
  }

  /*! \brief find value according data and size */
  entry_type find(const void * data, int size) const {
    uint64_t h1, h2;
    hasher_(data, size, &h1, &h2, 0);
    return search(h1, h2);
  }

  /*! \brief bi-model size */
  uint64_t size() const { return size_; }
  
  private:
    /*! \brief search record according h1 and h2 */
    entry_type search(uint64_t h1, uint64_t h2) const {
      // h1: find bucket index
      uint64_t bucket_idx = (h1 & bucket_1_);
      const_bucket_pointer bucket = 
        const_bucket_pointer(idx_ + bucket_idx);
      
      const_record_pointer begin = record_ + bucket->first,
                           end = record_ + bucket->second;
      // h2: sequential search
      while (begin < end) {
        key_type key = *const_key_pointer(begin);
        uint count = *(begin + key_size);
        if (key == h2) {
          const_value_pointer value = 
            const_value_pointer(begin + key_size + count_size);
          return std::make_pair(count, value);
        }
        begin += key_size + count_size + count * value_size;
      }
      const float * empty = NULL;
      return std::make_pair(0, empty);
    }
  private:
    /*! \brief bias coefficient */
    VType bias_coefficient_;
    /*! \brief length of latent vector */
    uint32_t embedding_size_;
    /*! \brief bucket size */
    idx_type bucket_1_;
    /*! \brief index region. bytes stored */
    const_idx_pointer idx_;
    /*! \brief record region. <key. count, value> */
    const_record_pointer record_;
    /*! \brief size of model bytes */
    uint64_t size_;
    /*! \brief hash mathod for generate index and key */
    typedef Hasher hasher_;

}; // struct FFMModel

} // namespace dstore
} // namespace mit
#endif // OPENMIT_TOOLS_DSTORE_FFM_TABLE_H_
