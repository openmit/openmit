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
#include "openmit/common/parameter/cli_param.h"
#include "openmit/tools/dstruct/dstring.h"

namespace mit {
/*! 
 * \brief w + (v11 v12 v13 v14 v21 v22 ...)
 */
struct Entry {
  /*! \brief constructor */
  Entry(const mit::CliParam & cli_param, size_t field_number = 0, mit_uint fieldid = 0l) {
    embedding_size = cli_param.embedding_size;
    // only w parameter. (lr || ffm that not cross field)
    if (cli_param.model == "lr" || field_number == 0) {
      length = 1;
      embedding_size = 0;
    } else if (cli_param.model == "fm") {
      length = 1 + embedding_size;
    } else if (cli_param.model == "ffm" && fieldid > 0l) {
      length = 1 + field_number * embedding_size;
      fieldid = fieldid;
    } else {
      LOG(FATAL) << "parameter error. model: " << cli_param.model 
        << ", field_number: " << field_number 
        << ", fieldid: " << fieldid; 
    }
    wv = new mit_float[length]();
    // init 
    for (auto i = 0u; i < length; ++i) { wv[i] = 0.0f; }
  }

  ~Entry() {
    if (wv) { delete[] wv; wv = nullptr; }
  }

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

  /*! \brief length of entry */
  size_t Size() const { return length; }

  /*! \brief field size */
  inline size_t field_number() const {
    return (length - 1) / embedding_size;
  }

  inline mit_float Get(size_t idx) {
    return *(wv + idx);
  }

  inline void Set(size_t idx, mit_float value) {
    *(wv + idx) = value;
  }

  inline mit_float * Data() { return wv; }

  inline mit_float GetW() { return *wv; }

  inline void SetW(const mit_float & weight) {
    *wv = weight;
  }

  /*! 
   * \brief i-th feild latent vector info 
   */
  inline mit_float * GetV(const size_t & index) {
    return wv + (1 + index * embedding_size);
  }

  inline void SetV(const size_t & index, 
                   const size_t & f, 
                   const mit_float & value) {
    auto offset = 1 + index * embedding_size + f;
    *(wv + offset) = value;
  }
};  // struct Entry

typedef std::unordered_map<mit_uint, mit::Entry * > PMAPT1;

} // namespace mit 
#endif // OPENMIT_ENTITY_ENTRY_H_
