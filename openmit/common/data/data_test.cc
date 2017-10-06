#include <string>
#include "openmit/common/data/data.h"
#include "dmlc/logging.h"
using namespace mit;

/*! information string of the row */
std::string RowInfo(const dmlc::Row<mit_uint> & row) {
  std::string line = std::to_string(row.get_label());
  for (auto i = 0u; i < row.length; ++i) {
    line += " " +  std::to_string(row.get_index(i)) + 
      ":" + std::to_string(row.get_value(i));
  }
  return line;
}

void MiniBatch(const dmlc::RowBlock<mit_uint> & batch) {
  LOG(INFO) << "batch.size: " << batch.size;
  for (auto i = 0u; i < batch.size; ++i) {
    LOG(INFO) << RowInfo(batch[i]);
  }
  LOG(INFO) << "feature number: " << (batch.offset[batch.size] - batch.offset[0]);
  for (auto i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
    LOG(INFO) << "batch.index[" << i << "] " << batch.index[i];
  }
}

int main(int argc, char ** argv) {
  char * file = argv[1];
  LOG(INFO) << "file: " << file;
  mit::DMatrix * data = new mit::DMatrix(std::string(file), 0, 1, "libsvm");
  
  int count = 0;
  data->BeforeFirst();
  while (data->Next()) {
    const auto & block = data->Value();
    count += block.size;
    int end = 0;
    for (auto i = 0u; i < block.size; i+=5) {
      LOG(INFO) << "============== MiniBatch. i: " << i << " ===============";
      end = i + 5 > block.size ? block.size : i+5;
      auto batch = block.Slice(i, end);
      MiniBatch(batch);
    }
  }
  return 0;
}
