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
  static Entry * Create(const mit::ModelParam & model_param, 
                        mit::EntryMeta * entry_meta, 
                        mit::math::Random * distr, 
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
  virtual std::string String(
    mit::EntryMeta * entry_meta = NULL) = 0;

  /*! \brief save entry */
  virtual void Save(dmlc::Stream * fo, 
    mit::EntryMeta * entry_meta = NULL) = 0;

  /*! \brief load entry */
  virtual void Load(dmlc::Stream * fi,
    mit::EntryMeta * entry_meta = NULL) = 0;

};  // struct Entry

/*! 
 * \brief general linear regression entry 
 */
struct LREntry : Entry {
  /*! \brief constructor */
  LREntry(const mit::ModelParam & model_param, 
          mit::EntryMeta * entry_meta,
          mit::math::Random * distr) {
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

  /*! \brief save entry */
  void Save(dmlc::Stream * fo, 
            mit::EntryMeta * entry_meta = NULL) override {
    fo->Write((char *) &wv[0], sizeof(mit_float));
  }

  /*! \brief load entry */
  void Load(dmlc::Stream * fi, 
            mit::EntryMeta * entry_meta = NULL) override {
    mit_float w;
    fi->Read(&w, sizeof(mit_float));
    wv[0] = w;
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
          mit::math::Random * distr) {
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

  /*! \brief save entry */
  void Save(dmlc::Stream * fo, 
            mit::EntryMeta * entry_meta = NULL) override {
    fo->Write((char *) &embedding_size, sizeof(size_t));
    fo->Write((char *) &wv[0], sizeof(mit_float));
    for (size_t k = 0; k < embedding_size; ++k) {
      fo->Write((char *) &wv[1 + k], sizeof(mit_float));
    }
  }

  /*! \brief load entry */
  void Load(dmlc::Stream * fi, 
            mit::EntryMeta * entry_meta = NULL) override {
    size_t embedsize; 
    fi->Read(&embedsize, sizeof(size_t));
    CHECK(embedding_size == embedsize && embedsize > 0);
    mit_float value;
    fi->Read(&value, sizeof(mit_float));
    wv[0] = value;
    for (size_t k = 0; k < embedsize; ++k) {
      fi->Read(&value, sizeof(mit_float));
      wv[1 + k] = value; 
    }
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
  FFMEntry(const mit::ModelParam& model_param, 
           mit::EntryMeta* entry_meta, 
           mit::math::Random* distr, 
           mit_uint field = 0l) {
    embedding_size = model_param.embedding_size;
    CHECK(embedding_size > 0) << "embedding_size error for fm/ffm model.";
    fieldid = field;
    length = 1;
    if (fieldid > 0) {
      std::vector<mit_uint>* rfields = entry_meta->CombineInfo(field);
      size_t rfield_cnt = 0;
      if (rfields) {
        rfield_cnt += rfields->size();
        length += rfield_cnt * embedding_size;
      }
    }
    wv = new mit_float[length]();
    for (auto idx = 0u; idx < length; ++idx) {
      wv[idx] = distr->random(); 
    }
  }

  /*! \brief destructor */
  ~FFMEntry() {}
  
  /*! \brief string format entry info */
  std::string String(mit::EntryMeta * entry_meta = NULL) override {
    CHECK_NOTNULL(entry_meta);
    std::string info = std::to_string(*wv);
    if (length == 1) return info;
    auto* rfields = entry_meta->CombineInfo(fieldid);
    if (!rfields) return info;
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
  
  /*! \brief save entry */
  void Save(dmlc::Stream* fo, mit::EntryMeta* entry_meta = NULL) override {
    fo->Write((char *) &fieldid, sizeof(mit_uint));
    fo->Write((char *) &embedding_size, sizeof(size_t));
    fo->Write((char *) &wv[0], sizeof(mit_float));
    auto* rfields = entry_meta->CombineInfo(fieldid);
    if (!rfields) return;
    for (auto i = 0u; i < rfields->size(); ++i) {
      auto offset_begin = 1 + i * embedding_size;
      mit_uint rfid = (*rfields)[i];
      fo->Write((char *) &rfid, sizeof(mit_uint));
      for (size_t k = 0; k < embedding_size; ++k) {
        mit_float v = wv[offset_begin + k];
        fo->Write((char *) &v, sizeof(mit_float));
      }
    }
  } // method Save

  /*! \brief save entry */
  void Load(dmlc::Stream * fi, 
            mit::EntryMeta * entry_meta = NULL) override {
    fi->Read(&fieldid, sizeof(mit_uint));
    fi->Read(&embedding_size, sizeof(size_t));
    fi->Read(&wv[0], sizeof(mit_float));
    auto* rfields = entry_meta->CombineInfo(fieldid);
    if (!rfields) return;
    size_t offset = 1;
    for (auto i = 0u; i < rfields->size(); ++i) {
      mit_uint rfid;
      fi->Read(&rfid, sizeof(mit_uint));
      CHECK_EQ(rfid, (*rfields)[i]) << "related fielid not match";
      for (size_t k = 0; k < embedding_size; ++k) {
        fi->Read(&wv[offset], sizeof(mit_float));
        offset++;
      }
    }
  } // method Load
}; // struct FFMEntry


/*! 
 * \brief factorization machine model store unit 
 */
struct MFEntry : Entry {
  /*! \brief length of latent factor */
  size_t embedding_size;

/*! \brief constructor for matrix factorization model */
  MFEntry(const mit::ModelParam & model_param,
          mit::EntryMeta * entry_meta,
          mit::math::Random* distr) {
    embedding_size = model_param.embedding_size;
    CHECK(embedding_size > 0)
      << "embedding_size should be > 0 for fm model.";
    length = embedding_size;
    wv = new mit_float[length]();
    for (auto idx = 0u; idx < length; ++idx) {
      wv[idx] = distr->random();
    }
  }

  /*! \brief destructor */
  ~MFEntry() {}

  /*! \brief string format entry info */
  std::string String(mit::EntryMeta * entry_meta = NULL) override {
    std::string info = std::to_string(wv[0]);
    for (auto k = 1u; k < embedding_size; ++k) {
      info += " " + std::to_string(wv[k]);
    }
    return info;
  }

  /*! \brief save entry */
  void Save(dmlc::Stream * fo,
            mit::EntryMeta * entry_meta = NULL) override {
    fo->Write((char *) &embedding_size, sizeof(size_t));
    for (size_t k = 0; k < embedding_size; ++k) {
      fo->Write((char *) &wv[k], sizeof(mit_float));
    }
  }

  /*! \brief load entry */
  void Load(dmlc::Stream * fi,
            mit::EntryMeta * entry_meta = NULL) override {
    size_t embedsize;
    fi->Read(&embedsize, sizeof(size_t));
    CHECK(embedding_size == embedsize && embedsize > 0);
    mit_float value;
    fi->Read(&value, sizeof(mit_float));
    wv[0] = value;
    for (size_t k = 0; k < embedsize; ++k) {
      fi->Read(&value, sizeof(mit_float));
      wv[1 + k] = value;
    }
  }
}; // struct MFEntry

// entry map type 
typedef std::unordered_map<mit_uint, mit::Entry * > entry_map_type;

} // namespace mit 
#endif // OPENMIT_ENTRY_ENTRY_H_
