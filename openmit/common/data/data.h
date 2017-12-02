/*!
 *  Copyright (c) 2016 by Contributors
 *  \file data.h
 *  \brief data instance set structure
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_DATA_DATA_H_
#define OPENMIT_COMMON_DATA_DATA_H_

#include <string>
#include "dmlc/data.h"
#include "dmlc/logging.h"
#include "openmit/common/base.h"

namespace mit {
/*!
 * \brief dataset structure 
 * \begincode
 *    std::string uri = "hdfs://uri1,hdfs://uri2,..."
 *    int partid = ps->MyRank();
 *    int npart = ps->NumWorkers();
 *    DMatrix * dm = new DMatrix(uri, partid, npart, "libsvm");
 *    dm->BeforeFirst();
 *    while (dm->Next()) {
 *      dmlc::RowBlock<mit_uint> & dblock = dm->Value();
 *      // computing based on dblock ....
 *    }
 * \endcode 
 */
class DMatrix {
  public:
    /*! \brief point to the initial position of data matrix */
    inline void BeforeFirst();
  
    /*! \biref determine whether there is a next data block */
    inline bool Next();

    /*! \brief fetch data block of current shifting */
    inline const dmlc::RowBlock<mit_uint> Value() { 
      return dmat_[index_]->Value(); 
    }
    
    /*! \return max key (feature dimension) of data matrix */
    inline uint64_t NumCol() const { return num_col_; }

    /*!
     * \brief constructor
     * \param uri data path 
              if there are multiple paths, the paths are separated by comma. such as: 
              hdfs data path: "hdfs://uri_path_1,hdfs://uri_path_2,..."
              local data path: "path1,path2,..."
     * \param partid id of current node that are involved in data computing
     * \param npart total data computing node numbers 
     * \param data_format it supports two format. "libsvm","libfm"
     * */
    DMatrix(const std::string& uri, int partid, int npart, std::string data_format) {
      if (data_format == "auto") data_format = "libsvm";
      std::vector<std::string> uri_items; 
      std::string uri_item; 
      std::istringstream is(uri);
      while (std::getline(is, uri_item, ',')) {
        CHECK(uri_item.size() > 0) << uri << " path error.";
        uri_items.push_back(uri_item);
      }
      CHECK(uri_items.size() > 0) << "path " << uri << " is empty!";

      num_col_ = 0l;
      for (auto i = 0u; i < uri_items.size(); ++i) {
        auto * dbi = dmlc::RowBlockIter<mit_uint>::Create(
          uri_items[i].c_str(), partid, npart, data_format.c_str());
        CHECK(dbi) << "dmlc::RowBlockIter is null";
        dmat_.push_back(dbi);
        num_col_ = num_col_ > dbi->NumCol() ? num_col_ : dbi->NumCol();
      }
      index_ = 0;
    }

    /*!
     * \brief destructor
     */
    ~DMatrix() { 
      for (auto idx = 0u; idx < dmat_.size(); ++idx) {
        if (dmat_[idx]) { delete dmat_[idx]; dmat_[idx] = NULL; }
      }
    }

  private:
    /*! \brief data storage structure. it supports multiple paths */
    std::vector<dmlc::RowBlockIter<mit_uint>* > dmat_;
    /*! \brief current data path offset index */
    size_t index_;
    /*! \brief max key of data matrix */
    uint64_t num_col_;
};

bool DMatrix::Next() {
  CHECK(index_ < dmat_.size());
  bool exist_next = true;
  if (dmat_[index_]->Next()) {
    return exist_next;
  } else if (index_ < dmat_.size() - 1 && !dmat_[index_]->Next()) {
    if (!dmat_[++index_]->Next()) exist_next = false;
  } else if (index_ == dmat_.size() -1 && !dmat_[index_]->Next()) {
    exist_next = false;
  }
  return exist_next;
}

void DMatrix::BeforeFirst() {
  index_ = 0;
  for (auto i = 0u; i < dmat_.size(); ++i) {
    dmat_[i]->BeforeFirst();
  }
}

} // namespace mit
#endif // OPENMIT_COMMON_DATA_DATA_H_
