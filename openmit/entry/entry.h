/*!
 *  Copyright (c) 2017 by Contributors
 *  \file entry.h
 *  \brief parameter computational factor that used to extract
 *        model interface and optimization algorithm interface
 *  \author ZhouYong
 */
#ifndef OPENMIT_ENTRY_ENTRY_H_
#define OPENMIT_ENTRY_ENTRY_H_ 

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <random>
#include <vector>
#include "openmit/common/base.h"
#include "openmit/common/parameter.h"
#include "openmit/entry/entry_meta.h"
#include "openmit/tools/dstruct/dstring.h"
#include "openmit/tools/math/random.h"

namespace mit {
/*! 
 * \brief entry model computational & store unit
 */
struct Entry {
  /*! \brief create a entry */
  static Entry* Create(const mit::ModelParam& model_param, 
                       mit::EntryMeta* entry_meta, 
                       mit::math::Random* distr, 
                       mit_uint field = 0l);

  virtual ~Entry() {
    if (wv) { delete[] wv; wv = nullptr; }
  }

  /*! \brief entry contents */
  mit_float* wv;
  
  /*! \brief length of model store unit */
  size_t length;

  /*! \brief length of entry */
  inline size_t Size() const { return length; }

  inline mit_float Get(size_t idx = 0) {
    return *(wv + idx);
  }

  inline void Set(size_t idx, mit_float value) {
    *(wv + idx) = value;
  }

  inline mit_float* Data() { return wv; }

  /*! \brief string format entry info */
  virtual std::string String(mit::EntryMeta* entry_meta = NULL) = 0;

  /*! \brief save entry */
  virtual void Save(dmlc::Stream* fo, mit::EntryMeta* entry_meta = NULL) = 0;

  /*! \brief load entry */
  virtual void Load(dmlc::Stream* fi, mit::EntryMeta* entry_meta = NULL) = 0;

};  // struct Entry

} // namespace mit 
#endif // OPENMIT_ENTRY_ENTRY_H_
