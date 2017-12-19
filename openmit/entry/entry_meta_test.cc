#include <string>
#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/parameter.h"
#include "openmit/entry/entry_meta.h"
using namespace mit;

template <typename V>
std::string VecInfo(std::vector<V> * vec) {
  if (vec->size() == 0) return "";
  std::string value = std::to_string(vec->at(0));
  for (size_t i = 1; i < vec->size(); ++i) {
    value += " " + std::to_string(vec->at(i));
  }
  return value;
}
int main(int argc, char ** argv) {
  CHECK_GE(argc, 2) << "Usage: " 
    << argv[0] << " openmit.conf [k1=v1] [k2=v2] ...";

  mit::ArgParser parser;
  if (strcmp(argv[1], "none")) parser.ReadFile(argv[1]);
  parser.ReadArgs(argc - 2, argv + 2);
  const mit::KWArgs kwargs = parser.GetKWArgs();

  mit::ModelParam model_param;
  model_param.InitAllowUnknown(kwargs);

  LOG(INFO) << "field_combine_set: " << model_param.field_combine_set;
  LOG(INFO) << "field_combine_pair: " << model_param.field_combine_pair;

  mit::EntryMeta * entry_meta = new mit::EntryMeta(model_param);
  auto * result = entry_meta->CombineInfo(3);
  LOG(INFO) << "fieldid=3, size: " << result->size();
  LOG(INFO) << VecInfo<mit_uint>(result);
  if (result->size() == 0) delete result;
  // test Save & Load 
  dmlc::Stream * fo = dmlc::Stream::Create(argv[2], "w");
  entry_meta->Save(fo);
  int value = 1001; fo->Write(&value, sizeof(int));
  delete fo;
  dmlc::Stream * fi = dmlc::Stream::Create(argv[2], "r");
  entry_meta->Load(fi);
  int xx; fi->Read(&xx, sizeof(int));
  LOG(INFO) << "load value: " << xx;
  delete fi;

  delete entry_meta;

  return 0;
}
