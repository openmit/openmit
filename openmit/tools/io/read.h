/*!
 *  Copyright 2016 by Contributors
 *  \file openmit/tools/io/read.h
 *  \brief io read operate
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_IO_READ_H_
#define OPENMIT_TOOLS_IO_READ_H_

#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <unistd.h>

namespace mit {
/*! \brief File IO Read Operator */
class Read {
  public:
    /*! \brief constructor by file and mode */
    explicit Read(const char * file, 
                  bool is_binary = false);
    /*! \brief destructor */
    ~Read() { close(); }

    /*! \brief open file road */
    bool open(const char * file, 
              bool is_binary = false);
    /*! \brief getline saved as string */
    bool get_line(std::string & line);
    /*! \brief check if a file is open */
    inline bool is_open() { return ifs_.is_open(); }
    /*! \brief close read operator */
    inline void close() { ifs_.close(); }

  private:
    /*! \brief input file stream */
    std::ifstream ifs_;
    /*! \brief whether binary mode */
    bool is_binary_;
}; // class Read

/*! 
 * \brief mmap based on memory read operator. 
 *        read data saved as void *. 
 */
class MMap {
  public:
    /*! \brief default constructor */
    MMap() : store_(0), size_(0) {}
    /*! \brief destructor using close method */
    ~MMap() { close(); }
    /*! \brief fetch data to store_ */
    bool load(const char * path);
    /*! \brief close data */
    void close();
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

} // namespace mit

#endif // OPENMIT_TOOLS_IO_READ_H_
