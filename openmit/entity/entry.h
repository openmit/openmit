/*!
 *  Copyright (c) 2017 by Contributors
 *  \file entry.h
 *  \brief parameter computational factor that used to extract
 *        model interface and optimization algorithm interface
 *  \author ZhouYong
 */
#ifndef OPENMIT_ENTITY_ENTRY_H_
#define OPENMIT_ENTITY_ENTRY_H_ 

#include <stdint.h>
#include <stddef.h>
#include <vector>
#include "openmit/common/base.h"
#include "openmit/tools/dstruct/dstring.h"

namespace mit {
/*! 
 * \brief w + (f1 v11 v12 v13 v14 f2 v21 v22 ...)
 */
struct Entry {
  /*! \brief constructor */
  Entry(mit_uint featid, 
        size_t embedding_size, 
        std::vector<mit_uint> fields, 
        mit_uint fieldid) :
  featid(featid), fieldid(fieldid), embedding_size(embedding_size) {
    length = 1 + fields.size() * (1 + embedding_size);
    wv = new mit_float[length]();
    for (auto i = 0u; i < fields.size(); ++i) {
      wv[1 + i * (1 + embedding_size)] = fields[i];
    }
  }

  /*! \brief feature key */
  mit_uint featid;
  /*! \brief feild key */
  mit_uint fieldid;
  /*! 
   * \brief entry information, it contains w and v 
   *  w: weight of feature; 
   *  v: latent vector of feature splited by field  
   */
  mit_float * wv;
  /*! \brief length of the (w + v) */
  size_t length;
  /*! \brief embedding size */
  size_t embedding_size;

  /*! 
   * \brief raw feature key 
   */
  inline mit_uint Key() const {
    return featid;
  }
  /*! \brief new feature key if libfm */
  inline mit_uint NewKey(size_t nbit) const {
    return (featid << nbit) + fieldid;
  }

  /*! 
   * \brief i-th feild latent vector info 
   */
  inline mit_float * GetV(const size_t & index) {
    return wv + (1 + index * (1 + embedding_size));
  }

  inline void SetV(const size_t & index, 
                   const size_t & f, 
                   const mit_float & value) {
    auto offset = 1 + index * (1 + embedding_size) + (f + 1);
    *(wv + offset) = value;
  }
}; 

} // namespace mit 

#endif // OPENMIT_ENTITY_ENTRY_H_
