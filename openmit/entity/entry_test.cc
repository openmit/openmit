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

  mit::CliParam cli_param;
  cli_param.InitAllowUnknown(kwargs);

  LOG(INFO) << "field_combine_set: " << cli_param.field_combine_set;
  LOG(INFO) << "field_combine_pair: " << cli_param.field_combine_pair;

  mit::EntryMeta * entry_meta = new mit::EntryMeta(cli_param);
  mit_uint fieldid = 3;
  auto * result = entry_meta->CombineInfo(fieldid);
  auto field_size = result->size();
  LOG(INFO) << "fieldid=3, size: " << field_size;
  LOG(INFO) << VecInfo<mit_uint>(result->data(), result->size());
  if (result->size() == 0) delete result;
  delete entry_meta;
  
  mit::Entry * entry = new mit::Entry(cli_param, field_size, fieldid);
  LOG(INFO) << "entry.length: " << entry->length;

  entry->SetW(0.01);
  entry->SetV(0, 0, 0.1234566);
  entry->SetV(1, 3, 0.3333333);
  LOG(INFO) << "entry content: " << VecInfo(entry->Data(), entry->length);

  delete entry;
  return 0;
}
