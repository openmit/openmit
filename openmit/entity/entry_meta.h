/*!
 *  Copyright (c) 2017 by Contributors
 *  \file entry.h
 *  \brief parameter computational factor meta information
 *  \author ZhouYong
 */
#ifndef OPENMIT_ENTITY_ENTRY_META_H_
#define OPENMIT_ENTITY_ENTRY_META_H_ 

#include <string>
#include <unordered_map>
#include <vector>
#include "openmit/common/base.h"
#include "openmit/common/parameter/cli_param.h"
#include "openmit/tools/dstruct/dstring.h"
#include "openmit/tools/util/type_conversion.h"

namespace mit {
/*!
 * \brief entry meta information 
 */
struct EntryMeta {
  /*! 
   * \brief map structure stored <fieldid, related_fields>
   */
  std::unordered_map<mit_uint, std::vector<mit_uint> * > fields_map;
  /*! \brief embedding_size for fm/ffm */
  size_t embedding_size;
  /*! \brief model */
  std::string model;

  /*! \brief constructor */
  EntryMeta(const mit::CliParam & cli_param);

  /*! \brief destructor */
  ~EntryMeta();

  /*! \brief field combine */
  std::vector<mit_uint> * CombineInfo(const mit_uint & fieldid);

  /*! \brief process field combine set */
  void ProcessFieldCombineSet(const std::string field_combine_set);

  /*! \brief process field combine pair */
  void ProcessFieldCombinePair(const std::string field_combine_pair);

  /*! \brief fill field info */
  void FillFieldInfo(mit_uint & field1, mit_uint & field2);
};

} // namespace mit 

#endif // OPENMIT_ENTITY_ENTRY_META_H_
