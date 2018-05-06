#include <string>
#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/parameter.h"
#include "openmit/entry/entry_meta.h"
using namespace mit;
#include "openmit/entry/entry_meta.pb.h"
using namespace mit::protobuf;

template <typename V>
std::string VecInfo(std::vector<V>* vec) {
  if (vec->size() == 0) return "";
  std::string value = std::to_string(vec->at(0));
  for (size_t i = 1; i < vec->size(); ++i) {
    value += " " + std::to_string(vec->at(i));
  }
  return value;
}
int main(int argc, char** argv) {
  CHECK_GE(argc, 3) << "Usage: " << argv[0] << " file_path openmit.conf [k1=v1] [k2=v2] ...";
  const char* file_path = argv[1];

  // args parse
  mit::ArgParser parser;
  const char* conf_file = argv[2];
  if (strcmp(conf_file, "none")) parser.ReadFile(conf_file);
  parser.ReadArgs(argc - 3, argv + 3);
  const mit::KWArgs kwargs = parser.GetKWArgs();

  // entry meta
  mit::ModelParam model_param;
  model_param.InitAllowUnknown(kwargs);
  LOG(INFO) << "field_combine_set: " << model_param.field_combine_set;

  mit::EntryMeta* entry_meta = new mit::EntryMeta(model_param);
  auto* result = entry_meta->CombineInfo(3);
  LOG(INFO) << "fieldid=3, related fieldid: " << VecInfo<mit_uint>(result);
  if (result->size() == 0) delete result;

  // Save & Load of entry_meta
  dmlc::Stream* fo = dmlc::Stream::Create(file_path, "w");
  entry_meta->Save(fo);
  int value = 1234; 
  fo->Write(&value, sizeof(int));
  delete fo;

  dmlc::Stream* fi = dmlc::Stream::Create(file_path, "r");
  entry_meta->Load(fi);
  int xx; fi->Read(&xx, sizeof(int));
  LOG(INFO) << "load value: " << xx;
  delete fi;

  delete entry_meta; 

  // entry_meta.proto
  mit::protobuf::FieldIdArray field_ids;
  field_ids.add_field_id_array(11);
  field_ids.add_field_id_array(12);
  field_ids.add_field_id_array(13);
  field_ids.add_field_id_array(14);
  field_ids.add_field_id_array(15);  // bytes: 1
  field_ids.add_field_id_array(10000);  // bytes: 2
  LOG(INFO) << "fieldids.field_id_array(2): " << field_ids.field_id_array(2);   // 13
  LOG(INFO) << "fieldids.field_id_array_size: " << field_ids.field_id_array_size();  // 6
  LOG(INFO) << "field_ids.byte_size: " << field_ids.ByteSizeLong();  // 2 + (5*1 + 1*2)

  // CopyFrom
  mit::protobuf::FieldIdArray field_ids2;
  field_ids2.CopyFrom(field_ids);
  LOG(INFO) << "[CopyFrom] fieldids2.field_id_array(2): " << field_ids2.field_id_array(2);   // 13
  LOG(INFO) << "\tfieldids2.field_id_array_size: " << field_ids2.field_id_array_size();  // 6
  LOG(INFO) << "\tfield_ids2.byte_size: " << field_ids2.ByteSizeLong(); // 9

  // MergeFrom 
  FieldIdArray field_ids_merge;
  field_ids_merge.MergeFrom(field_ids);
  field_ids_merge.MergeFrom(field_ids2);
  field_ids_merge.set_field_id_array(2, 2222);
  LOG(INFO) << "[MergeFrom] fieldids_merge.field_id_array(2): " << field_ids_merge.field_id_array(2);   // 2222
  LOG(INFO) << "\tfieldids_merge.field_id_array_size: " << field_ids_merge.field_id_array_size();  // 12
  LOG(INFO) << "\tfield_ids_merge.byte_size: " << field_ids_merge.ByteSizeLong(); // 10 + (5*1 + 1*2) 

  // GetCacheSize
  LOG(INFO) << "[GetCacheSize] field_ids_merge.GetCachedSize(): " << field_ids_merge.GetCachedSize();

  mit::protobuf::EntryMeta entry_meta_pb;
  entry_meta_pb.set_embedding_size(8);
  entry_meta_pb.set_model("model_name");
  std::string key = "pb_key";
  auto mutable_map = entry_meta_pb.mutable_entry_meta_map();
  (*mutable_map)[key] = field_ids_merge;
  LOG(INFO) << "[EntryMeta] model: " << entry_meta_pb.model();
  LOG(INFO) << "\tentry_meta_map: " << entry_meta_pb.mutable_entry_meta_map();
  LOG(INFO) << "\tentry_meta_map.size: " << entry_meta_pb.entry_meta_map_size();
  LOG(INFO) << "\tentry_meta_map.byte_size: " << entry_meta_pb.ByteSizeLong();

  return 0;
}
