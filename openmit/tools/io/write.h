/*!
 *  Copyright 2016 by Contributors
 *  \file openmit/tools/io/write.h
 *  \brief io write operate
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_IO_WRITE_H_
#define OPENMIT_TOOLS_IO_WRITE_H_

#include <fstream>
#include <iostream>
using namespace std;

namespace mit {
/*! 
 * \brief File IO Write Operator 
 */
class Write {
  public:
    /*! \brief constructor by file and mode */
    explicit Write(const char * file,
                   bool is_binary = false,
                   bool is_append = false);
    
    /*! \brief destructor */
    ~Write() { close(); }
    
    /*! \brief open file road */
    bool open(const char * file, 
              std::ios_base::openmode mode);
    
    /*! \brief write text data, add linefeeds */
    bool write_line(const char * content);
    
    /*! \brief write text data */
    bool write(const char * content, 
               bool endln = false);
    
    /*! \brief write binary data */
    bool write_binary(const char * content, size_t size);

    /*! \brief check if a file is open */
    inline bool is_open() { return ofs_.is_open(); }
    
    /*! \brief close read operator */
    inline void close() { ofs_.close(); }

    inline bool is_binary() const { return is_binary_; }

  private:
    /*! \brief out file stream */
    std::ofstream ofs_;
    /*! \brief whether binary output */
    bool is_binary_;
}; // class Write
} // namespace mit
 
#endif // OPENMIT_TOOLS_IO_WRITE_H_
