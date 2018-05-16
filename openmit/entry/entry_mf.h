/*!
 *  Copyright (c) 2017 by Contributors
 *  \file entry_mf.h
 *  \brief mf model store unit
 *  \author ZhouYong
 */
#ifndef OPENMIT_ENTRY_ENTRY_MF_H_
#define OPENMIT_ENTRY_ENTRY_MF_H_ 

#include "openmit/entry/entry.h"

namespace mit {
/*! 
 * \brief matrix factorization model store unit 
 */
struct MFEntry : Entry {
  /*! \brief length of latent vector */
  size_t embedding_size;

  /*! \brief constructor */
  MFEntry(const mit::ModelParam& model_param, 
          mit::EntryMeta* entry_meta, 
          mit::math::Random* random);

  /*! \brief destructor */
  ~MFEntry() {}

  /*! \brief string format entry info */
  std::string String(mit::EntryMeta* entry_meta = NULL) override;
  
  /*! \brief save entry */
  void Save(dmlc::Stream* fo, mit::EntryMeta* entry_meta = NULL) override;

  /*! \brief save entry */
  void Load(dmlc::Stream* fi, mit::EntryMeta* entry_meta = NULL) override;
}; // struct MFEntry

MFEntry::MFEntry(const mit::ModelParam& model_param, 
                 mit::EntryMeta* entry_meta, 
                 mit::math::Random* random) {
  embedding_size = model_param.embedding_size;
  CHECK(embedding_size > 0 ) << "embedding_size set error.";
  length = embedding_size;
  wv = new mit_float[length]();
  for (auto idx = 0u; idx < length; ++idx) {
    wv[idx] = random->random();
  }
} // MFEntry

std::string MFEntry::String(mit::EntryMeta* entry_meta) {
  std::string info = "";
  for (auto k = 0u; k < embedding_size; ++k) {
    if (k == 0u) {
      info += std::to_string(wv[k]);
    }
    else {
      info += " " + std::to_string(wv[k]);
    }
  }
  return info;
} // MFEntry::String 

void MFEntry::Save(dmlc::Stream* fo, mit::EntryMeta* entry_meta) {
  fo->Write((char*)&embedding_size, sizeof(size_t));
  for (size_t k = 0; k < embedding_size; ++k) {
    fo->Write((char*)&wv[k], sizeof(mit_float));
  }
} // MFEntry::Save 

void MFEntry::Load(dmlc::Stream* fi, mit::EntryMeta* entry_meta) {
  size_t embedsize;
  fi->Read(&embedsize, sizeof(size_t));
  CHECK(embedding_size == embedsize && embedsize > 0);
  mit_float value;
  for (size_t k = 0; k < embedsize; ++k) {
    fi->Read(&value, sizeof(mit_float));
    wv[k] = value;
  }
} // MFEntry::Load

}// namespace mit
#endif // OPENMIT_ENTRY_ENTRY_MF_H_
