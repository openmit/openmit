/*!
 *  Copyright 2016 by Contributors
 *  \file data.h
 *  \brief io read operate
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_IO_READ_H_
#define OPENMIT_TOOLS_IO_READ_H_

#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <unistd.h>

namespace mit {
/*! \brief mmap read data saved as void *. */
class MMap {
  public:
    /*! \brief default constructor */
    MMap() : store_(0), size_(0) {}
    /*! \brief destructor using close method */
    ~MMap() { close(); }
    /*! \brief fetch data to store_ */
    inline bool load(const char * path);
    /*! \brief close data */
    inline void close();
    /*! \brief get data content */
    inline void * data() const { return store_; }
    /*! \brief data size */
    inline uint64_t size() const { return size_; }

  private:
    /*! \brief store data content */
    void * store_;
    /*! \brief data size */
    uint64_t size_;

}; // class MMap

inline bool MMap::load(const char * path) {
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

inline void MMap::close() {
  if (::munmap(store_, size_) < 0) {
    ::err(errno, "MMap munmap failed.");
  }
  store_ = 0;
  size_ = 0;
}


/*! \brief IO Read Operator */
class Read {

}; // Read
} // namespace mit

#endif // OPENMIT_TOOLS_IO_READ_H_
