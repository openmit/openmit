/*!
 *  Copyright (c) 2017 by Contributors
 *  \file entry_ffm.h
 *  \brief ffm model store unit
 *  \author ZhouYong
 */
#ifndef OPENMIT_ENTRY_ENTRY_FFM_H_
#define OPENMIT_ENTRY_ENTRY_FFM_H_ 

#include "openmit/entry/entry.h"

namespace mit { 
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
           mit::math::Random* random, 
           mit_uint field = 0l);

  /*! \brief destructor */
  ~FFMEntry() {}
  
  /*! \brief string format entry info */
  std::string String(mit::EntryMeta* entry_meta = NULL) override;
  
  /*! \brief save entry */
  void Save(dmlc::Stream* fo, mit::EntryMeta* entry_meta = NULL) override;

  /*! \brief save entry */
  void Load(dmlc::Stream* fi, mit::EntryMeta* entry_meta = NULL) override;
}; // struct FFMEntry


FFMEntry::FFMEntry(const mit::ModelParam& model_param, 
                   mit::EntryMeta* entry_meta, 
                   mit::math::Random* random, 
                   mit_uint field) {
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
    wv[idx] = random->random(); 
  }
} // FFMEntry

std::string FFMEntry::String(mit::EntryMeta* entry_meta) {
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
} // FFMEntry::String 

void FFMEntry::Save(dmlc::Stream* fo, mit::EntryMeta* entry_meta) {
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
} // FFMEntry::Save

void FFMEntry::Load(dmlc::Stream* fi, mit::EntryMeta* entry_meta) {
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
} // FFMEntry::Load

} // namespace mit
#endif // OPENMIT_ENTRY_ENTRY_FFM_H_
