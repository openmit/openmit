#include <fstream>
#include <iostream>
#include <stdint.h>
#include "openmit/tools/io/write.h"
#include "openmit/tools/io/read.h"
using namespace std;
 
void write(ofstream & ofs, const char * content) {
  ofs << content << endl;
}

int main(int argc, char ** argv) {
  /*
  // [test] write text data
  std::string str = "I love this world!!!!";
  mit::Write write(argv[1]);
  write.write_line(str.c_str());
  */
  // [test] write binary data
  mit::Write obin(argv[1], true);
  uint32_t value_num = 5;
  float bias = 0.123456;
  std::cout << "write value_num: " << value_num << std::endl;
  std::cout << "write bias: " << bias << std::endl;
  obin.write_binary((char *) &value_num, sizeof(uint32_t));
  obin.write_binary((char *) &bias, sizeof(float));
  obin.close();

  mit::MMap mmap;
  mmap.load(argv[1]);
  const char * data = reinterpret_cast<const char *>(mmap.data());
  std::cout << "read value_num: " << *(uint32_t *) data << std::endl;
  std::cout << "read bias: " << *(float *) (data + sizeof(uint32_t)) << std::endl;
  return 0;
}
