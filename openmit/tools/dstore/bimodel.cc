#include "openmit/tools/io/read.h"
#include "openmit/tools/io/write.h"
#include "openmit/tools/hash/murmur3.h"
#include "openmit/tools/dstruct/dstring.h"
#include "openmit/tools/util/type_conversion.h"
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdint.h>
using namespace std;

class Record {
public:
  Record(std::string value) : value_info_(value) {
    Init();
  }

  void Init() {
    std::vector<std::string> kv;
    mit::string::Split(value_info_, &kv, '#');
    if (kv.size() != 2) { return ; }
    key_ = mit::StringToNum<uint64_t>(kv[0]);

    std::vector<std::string> value_items;
    mit::string::Split(kv[1], &value_items, ',');
    value_num_ = value_items.size();
    if (value_items.size() == 0) {
      std::cerr << "record value_items.size == 0" << std::endl;
    }
    std::vector<float> * values = new std::vector<float>();
    for (auto i = 0u; i < value_num_; ++i) {
      values->push_back(mit::StringToNum<float>(value_items[i]));
    }
    value_ = values->data();
  }

  ~Record() { Clear(); }

  void Write(mit::Write * write) {
    write->write_binary((char *) &key_, sizeof(uint64_t));
    for (auto i = 0u; i < value_num_; ++i) {
      float value = *(value_ + i);
      write->write_binary((char *) &value, sizeof(float));
    }
  }

  std::string ValueInfo() const { return value_info_; }

  void Clear() {
    if (value_) { delete value_; value_ = NULL; }
  }
private:
    std::string value_info_;
    uint64_t key_;
    float * value_;
    uint32_t value_num_;


}; // class Record

int main(int argc, char ** argv) {
  typedef mit::hash::MMHash128 hasher_;
  uint32_t bucket_size = 64;
  const char * infile = argv[1];
  const char * outfile = argv[2];
  mit::Read read(infile, false);
  std::string line;
  std::vector<std::string> line_items;
  std::vector<Record *> * record_table[bucket_size];
  for (uint32_t i = 0; i < 64; ++i) {
    record_table[i] = new std::vector<Record *>();
  }

  while(read.get_line(line)) {
    mit::string::Split(line, &line_items, '\t');
    if (line_items.empty() || line_items.size() != 2) 
      continue;
    std::string key = line_items[0];
    std::string value = line_items[1];
    uint64_t h1, h2;
    hasher_(key.c_str(), key.size(), &h1, &h2, 0);
    uint32_t index = h1 & (bucket_size - 1);
    std::string value_info = std::to_string(h2) + "#" + value;
    Record * record = new Record(value_info);
    record_table[index]->push_back(record);
  }

  mit::Write * write = new mit::Write(outfile, true);
  float coefficient = 0.999999;
  write->write_binary((char *) &coefficient, sizeof(float));

  uint32_t value_num = 1;
  write->write_binary((char *) &value_num, sizeof(uint32_t));
  // index region
  write->write_binary((char *) &bucket_size, sizeof(uint32_t));
  uint32_t index_offset = 0;
  write->write_binary((char *) &index_offset, sizeof(uint32_t));
  for (auto i = 0u; i < bucket_size; ++i) {
    index_offset += record_table[i]->size();
    write->write_binary((char *) &index_offset, sizeof(uint32_t));
  }
  // record region
  for (auto i = 0u; i < bucket_size; ++i) {
    std::vector<Record *> * records = record_table[i];
    for (auto j = 0u; j < records->size(); ++j) {
      (*records)[j]->Write(write);
    }
  }
  write->close();
  return 0;
}
