#include "openmit/common/arg.h"
#include "openmit/common/parameter/cli_param.h"
#include "openmit/entity/entry_meta.h"
#include "openmit/common/base.h"
using namespace mit;
#include <string>

template <typename V>
std::string VecInfo(std::vector<V> * vec) {
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

  mit::CliParam cli_param;
  cli_param.InitAllowUnknown(kwargs);

  LOG(INFO) << "field_combine_set: " << cli_param.field_combine_set;
  LOG(INFO) << "field_combine_pair: " << cli_param.field_combine_pair;

  mit::EntryMeta * entry_meta = new mit::EntryMeta(cli_param);
  auto * result = entry_meta->CombineInfo(3);
  LOG(INFO) << "fieldid=3, size: " << result->size();
  LOG(INFO) << VecInfo<mit_uint>(result);
  if (result->size() == 0) delete result;
  delete entry_meta;

  return 0;
}
