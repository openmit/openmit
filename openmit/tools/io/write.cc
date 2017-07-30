#include "openmit/tools/io/write.h"

namespace mit {

Write::Write(const char * file, 
             bool is_binary,
             bool is_append) {
  is_binary_ = is_binary;
  std::ios_base::openmode mode = 
    is_append ? std::ios_base::app : std::ios_base::trunc;
  mode = is_binary_ ? (mode | std::ios_base::binary) : mode;
  std::cout << "mode: " << mode << std::endl;
  open(file, mode);
}

bool Write::open(const char * file, 
                 std::ios_base::openmode mode) {
  ofs_.open(file, mode);
  return ofs_.is_open() && ofs_.good();
}

bool Write::write_line(const char * content) {
  // CHECK
  if (is_binary_) {
    std::cout << "write_line function not suitable to write binary file." << std::endl;
    return false;
  }
  return write(content, true);
}

bool Write::write(const char * content, bool endln) {
  if (is_binary_) {
    std::cout << "write_line function not suitable to write binary file." << std::endl;
    return false;
  }
  if (ofs_.good()) {
    ofs_ << content;
    if (endln) ofs_ << std::endl;
    return true;
  }
  return false;
}

bool Write::write_binary(const char * content, size_t size) {
  if (! is_binary_) {
    std::cout << "current out fstream is not ios_base::binary mode." << std::endl;
    return false;
  }
  ofs_.write(content, size);
}

} // namespace
