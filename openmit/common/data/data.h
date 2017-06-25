/*!
 *  Copyright (c) 2016 by Contributors
 *  \file data.h
 *  \brief stored instance data structure
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_DATA_H_
#define OPENMIT_COMMON_DATA_H_

#include <string>

#include "dmlc/data.h"
#include "dmlc/logging.h"
#include "openmit/common/base.h"

namespace mit {
/*! \brief define dataset structure */
class DMatrix {
  public:
    /*!
     * \brief constructor
     * \param uri data path
     * \param row_split whether to split data according to worker number
     * \param data_format "auto","libsvm", "libfm"
     * */
    DMatrix(const std::string & uri,
            int partid,
            int npart,
            std::string data_format) {
      Load(uri, partid, npart, data_format);
    }

    /*! \brief load method that be suitable for local and yarn environment */
    inline void Load(const std::string & uri,
                     int partid,
                     int npart,
                     std::string & data_format);

    /*! \brief before first  */
    void BeforeFirst() { dmat_->BeforeFirst(); }
    /*! \brief next  */
    bool Next() { return dmat_->Next(); }
    /*! \brief get block data  */
    const dmlc::RowBlock<mit_uint> Value() { return dmat_->Value(); }

    /*! \return size of bytes read so far */
    size_t ByteRead(void) const;
    /*! \return maximum feature dimension in the dataset */
    size_t NumCol() const { return dmat_->NumCol(); };

  private:
    /*! \brief data stored structure.
     *        Note: dmlc::Parser have not Numcol() method */
    dmlc::RowBlockIter<mit_uint> * dmat_;

}; // class DMatrix

inline void DMatrix
::Load(const std::string & uri,
       int partid,
       int npart,
       std::string & data_format) {
  if (data_format == "auto") data_format = "libsvm";
  dmat_ = dmlc::RowBlockIter<mit_uint>::Create(
      uri.c_str(), partid, npart, data_format.c_str());
}

} // namespace mit

#endif // OPENMIT_COMMON_DATA_H_
