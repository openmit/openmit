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
#include <string>
#include <random>
#include <vector>
#include "openmit/common/base.h"
#include "openmit/common/parameter/model_param.h"
#include "openmit/entity/entry_meta.h"
#include "openmit/tools/dstruct/dstring.h"
#include "openmit/tools/math/prob_distr.h"

namespace mit {
/*! 
 * \brief entry model computational & store unit
 */
struct Entry {
  /*! \brief create a entry */
  static Entry * Create(const mit::ModelParam & model_param, 
                        mit::EntryMeta * entry_meta, 
                        mit::math::ProbDistr * distr, 
                        mit_uint field = 0l);

  virtual ~Entry() {
    if (wv) { delete[] wv; wv = nullptr; }
  }

  /*! 
   * \brief entry information, it contains w and embedding factor
   */
  mit_float * wv;
  
  /*! \brief length of the w and embedding factor */
  size_t length;

  /*! \brief length of entry */
  inline size_t Size() const { return length; }

  inline mit_float Get(size_t idx = 0) {
    return *(wv + idx);
  }

  inline void Set(size_t idx, mit_float value) {
    *(wv + idx) = value;
  }

  inline mit_float * Data() { return wv; }

  /*! \brief string format entry info */
  virtual std::string String(
    mit::EntryMeta * entry_meta = NULL) = 0;

};  // struct Entry

/*! 
 * \brief general linear regression entry 
 */
struct LREntry : Entry {
  /*! \brief constructor */
  LREntry(const mit::ModelParam & model_param, 
          mit::EntryMeta * entry_meta,
          mit::math::ProbDistr * distr) {
    length = 1;
    wv = new mit_float[length]();
    wv[0] = distr->random();
  }

  /*! \brief destructor */
  ~LREntry() {}

  /*! \brief string format entry info */
  std::string String(mit::EntryMeta * entry_meta = NULL) override {
    return std::to_string(*wv);
  }
}; // struct LREntry

/*! 
 * \brief factorization machine model store unit 
 */
struct FMEntry : Entry {
  /*! \brief length of latent factor */
  size_t embedding_size;

  /*! \brief constructor */
  FMEntry(const mit::ModelParam & model_param, 
          mit::EntryMeta * entry_meta, 
          mit::math::ProbDistr * distr) {
    embedding_size = model_param.embedding_size;
    CHECK(embedding_size > 0) 
      << "embedding_size should be > 0 for fm model.";
    length = 1 + embedding_size;
    wv = new mit_float[length]();
    for (auto idx = 0u; idx < length; ++idx) {
      wv[idx] = distr->random();
    }
  }

  /*! \brief destructor */
  ~FMEntry() {}

  /*! \brief string format entry info */
  std::string String(mit::EntryMeta * entry_meta = NULL) override {
    std::string info = std::to_string(wv[0]);
    for (auto k = 0u; k < embedding_size; ++k) {
      info += " " + std::to_string(wv[1 + k]);
    }
    return info;
  }
}; // struct FMEntry

/*
 * \brief field-awared factorization machine model store unit 
 */
struct FFMEntry : Entry {
  /*! \brief length of latent vector */
  size_t embedding_size;

  /*! \brief field id */
  mit_uint fieldid;

  /*! \brief constructor */
  FFMEntry(const mit::ModelParam & model_param, 
           mit::EntryMeta * entry_meta, 
           mit::math::ProbDistr * distr, 
           mit_uint field = 0l) {
    embedding_size = model_param.embedding_size;
    CHECK(embedding_size > 0) 
      << "embedding_size should be > 0 for fm model.";
    fieldid = field;
    length = 1;
    if (fieldid > 0l) {
      auto rfield_cnt = entry_meta->CombineInfo(field)->size();
      length += rfield_cnt * embedding_size;
    }
    wv = new mit_float[length]();
    for (auto idx = 0u; idx < length; ++idx) {
      wv[idx] = distr->random(); 
      LOG(INFO) << "wv[" << idx << "]: " << wv[idx];
    }
  }

  /*! \brief destructor */
  ~FFMEntry() {}
  
  /*! \brief string format entry info */
  std::string String(mit::EntryMeta * entry_meta = NULL) override {
    std::string info = std::to_string(*wv);
    if (length == 1) return info;
    auto * rfields = entry_meta->CombineInfo(fieldid);
    for (auto i = 0u; i < rfields->size(); ++i) {
      auto idx_begin = 1 + i * embedding_size;
      info += " " + std::to_string((*rfields)[i]);
      info += ":" + std::to_string(wv[idx_begin + 0]);
      for (auto k = 1u; k < embedding_size; ++k) {
        info += "," + std::to_string(wv[idx_begin + k]);
      }
    }
    return info;
  } 
}; // struct FFMEntry

// entry map type 
typedef std::unordered_map<mit_uint, mit::Entry * > entry_map_type;

} // namespace mit 
#endif // OPENMIT_ENTITY_ENTRY_H_
