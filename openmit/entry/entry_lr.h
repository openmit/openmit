/*!
 *  Copyright (c) 2017 by Contributors
 *  \file entry_lr.h
 *  \brief linear model store unit
 *  \author ZhouYong
 */
#ifndef OPENMIT_ENTRY_ENTRY_LR_H_
#define OPENMIT_ENTRY_ENTRY_LR_H_ 

#include "openmit/entry/entry.h"

namespace mit {
/*! 
 * \brief general linear regression model store unit
 */
struct LREntry : Entry {
  /*! \brief constructor */
  LREntry(const mit::ModelParam& model_param, 
          mit::EntryMeta* entry_meta,
          mit::math::Random* random) {
    length = 1;
    wv = new mit_float[length]();
    wv[0] = random->random();
  }

  /*! \brief destructor */
  ~LREntry() {}

  /*! \brief string format entry info */
  inline std::string String(mit::EntryMeta* entry_meta = NULL) override {
    return std::to_string(*wv);
  }

  /*! \brief save entry */
  void Save(dmlc::Stream* fo, mit::EntryMeta* entry_meta = NULL) override {
    fo->Write((char*) &wv[0], sizeof(mit_float));
  }

  /*! \brief load entry */
  void Load(dmlc::Stream* fi, mit::EntryMeta* entry_meta = NULL) override {
    mit_float w;
    fi->Read(&w, sizeof(mit_float));
    wv[0] = w;
  }
}; // struct LREntry 

} // namespace mit
#endif // OPENMIT_ENTRY_ENTRY_LR_H_ 
