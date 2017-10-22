#include "openmit/common/arg.h"
#include "openmit/entity/entry.h"
#include "openmit/entity/entry_meta.h"
using namespace mit;
#include <string>

template <typename V>
std::string VecInfo(V * vec, size_t size) {
  std::string value = std::to_string(*(vec + 0));
  for (size_t i = 1; i < size; ++i) {
    value += " " + std::to_string(*(vec + i));
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
  //mit::EntryMeta * entry_meta = new mit::EntryMeta(model_param);
  std::unique_ptr<mit::EntryMeta> entry_meta;
  entry_meta.reset(new mit::EntryMeta(model_param));
  mit_uint fieldid = 3;
  auto * result = entry_meta->CombineInfo(fieldid);
  auto field_number = result->size();
  LOG(INFO) << "fieldid=3, size: " << field_number;
  LOG(INFO) << VecInfo<mit_uint>(result->data(), result->size());
  if (result->size() == 0) delete result;
  
  mit::math::ProbDistr * distr = mit::math::ProbDistr::Create(model_param);
  mit::Entry * entry = mit::Entry::Create(model_param, entry_meta.get(), distr, fieldid);
  LOG(INFO) << "entry.length: " << entry->length;

  
  entry->Set(0, 0.01);
  LOG(INFO) << "entry content: " << VecInfo(entry->Data(), entry->length);

  delete entry;
  return 0;
}
