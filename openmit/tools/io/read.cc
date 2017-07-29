#include "openmit/tools/io/read.h"

namespace mit {
// Read
bool Read::open(const char * file,
                std::ios_base::openmode mode) {
  ifs_.open(file, mode);
  return is_open() && ifs_.good();
}

bool Read::get_line(std::string & line) {
  return ifs_.good() && std::getline(ifs_, line);
  // return !ifs_.eof() && !ifs_.fail() && std::getline(ifs_, line);
}

// MMap
bool MMap::load(const char * path) {
  int fd = ::open(path, O_RDONLY);
  if (fd < 0) {
    ::err(errno, "MMap open failed. file: %s", path);
    return false;
  }
  struct stat sb;
  if (::fstat(fd, &sb) < 0) { 
    ::err(errno, "MMap fstst failed. file: %s", path);
    return false;
  }
  size_ = sb.st_size;
  
  store_ = ::mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  if (store_ == MAP_FAILED) {
    ::err(errno, "MMap map failed. file: %s", path);
    return false;
  }
  if (::close(fd) < 0) {
    ::err(errno, "MMap close failed. file: %s", path);
    return false;
  }
  return true;
}

void MMap::close() {
  if (::munmap(store_, size_) < 0) {
    ::err(errno, "MMap munmap failed.");
  }
  store_ = 0;
  size_ = 0;
}

} // namespace mit
