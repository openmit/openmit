#include "openmit/tools/io/write.h"

namespace mit {

Write::Write(const char * file, bool is_append) {
  std::ios_base::openmode mode = 
    is_append ? std::ios_base::app : std::ios_base::trunc;
  open(file, mode);
}

bool Write::open(const char * file, 
                 std::ios_base::openmode mode) {
  ofs_.open(file, mode);
  return ofs_.is_open() && ofs_.good();
}

bool Write::write_line(const char * content) {
  return write(content, true);
}

bool Write::write(const char * content, bool endln) {
  if (ofs_.good()) {
    ofs_ << content;
    if (endln) ofs_ << std::endl;
    return true;
  }
  return false;
}

} // namespace
