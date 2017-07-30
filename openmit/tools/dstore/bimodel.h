/*!
 *  Copyrigh 2017 by Contributors
 *  \file openmit/tools/dstore/bimodel.h
 *  \brief memory-based binary model file generation.
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_DSTORE_BIMODEL_H_
#define OPENMIT_TOOLS_DSTORE_BIMODEL_H_

#include <string>
#include "openmit/tools/hash/murmur3.h"
#include "openmit/tools/io/read.h"
#include "openmit/tools/io/write.h"

namespace mit {
namespace dstore {

/*! 
 * \brief binary model generate 
 */
template <typename VType, typename Hasher = mit::hash::MMHash128>
class BiModel {
  public:
    BiModel(uint32_t bucket_size) : bucket_size_(bucket_size) {}
    ~BiModel() {}
    /*!
     * \brief generate binary model file
     * \param infile model text file
     * \param outfile model binary file
     */
    bool GenerateModel(const char * infile, const char * outfile);

    /*!
     * \brief indexing 
     */
    bool IndexRegion();
    /*! 
     * \brief record
     */
    bool RecordRegion();

    /*!
     * \brief merge index & data 
     */
    bool MergeModel();

  private:
    ModelText(mit::Read & read);


  private:
    Hasher hasher_;
    uint32_t bucket_size;
    mit::Read read_;
    mit::Write write_;

}

} // namespace dstore
} // namespace mit

#endif // OPENMIT_TOOLS_DSTORE_BIMODEL_H_
