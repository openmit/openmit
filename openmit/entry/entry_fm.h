/*!
 *  Copyright (c) 2017 by Contributors
 *  \file entry_fm.h
 *  \brief fm model store unit
 *  \author WangYongJie
 */
#ifndef OPENMIT_ENTRY_ENTRY_FM_H_
#define OPENMIT_ENTRY_ENTRY_FM_H_ 

#include "openmit/entry/entry.h"

namespace mit {
/*! 
 * \brief factorization machine model store unit 
 */
struct FMEntry : mit::Entry {
  /*! \brief length of latent factor */
  size_t embedding_size;

  /*! \brief constructor */
  FMEntry(const mit::ModelParam& model_param, 
          mit::EntryMeta* entry_meta, 
          mit::math::Random* random);

  /*! \brief destructor */
  ~FMEntry() {}

  /*! \brief string format entry info */
  std::string String(mit::EntryMeta* entry_meta = NULL) override;

  /*! \brief save entry */
  void Save(dmlc::Stream* fo, mit::EntryMeta* entry_meta = NULL) override;

  /*! \brief load entry */
  void Load(dmlc::Stream* fi, mit::EntryMeta* entry_meta = NULL) override;
}; // struct FMEntry

FMEntry::FMEntry(const mit::ModelParam& model_param, 
                 mit::EntryMeta* entry_meta, 
                 mit::math::Random* random) {
  embedding_size = model_param.embedding_size;
  CHECK(embedding_size > 0) << "embedding_size set error.";
  length = 1 + embedding_size;
  wv = new mit_float[length]();
  for (auto idx = 0u; idx < length; ++idx) {
    wv[idx] = random->random();
  }
} // FMEntry

std::string FMEntry::String(mit::EntryMeta* entry_meta) {
  std::string info = std::to_string(wv[0]);
  for (auto k = 0u; k < embedding_size; ++k) {
    info += " " + std::to_string(wv[1 + k]);
  }
  return info;
} // FMEntry::String

void FMEntry::Save(dmlc::Stream* fo, mit::EntryMeta* entry_meta) {
  fo->Write((char *) &embedding_size, sizeof(size_t));
  fo->Write((char *) &wv[0], sizeof(mit_float));
  for (size_t k = 0; k < embedding_size; ++k) {
    fo->Write((char *) &wv[1 + k], sizeof(mit_float));
  }
} // FMEntry::Save

void FMEntry::Load(dmlc::Stream* fi, mit::EntryMeta* entry_meta) {
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
} // FMEntry::Load

} // namespace mit
#endif // OPENMIT_ENTRY_ENTRY_FM_H_
