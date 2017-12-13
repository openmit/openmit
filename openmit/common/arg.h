/*!
 *  Copyright (c) 2016 by Contributors
 *  \file base.h
 *  \brief argument parser for configuration file
 *  \author ZhouYong
 */
#ifndef OPENMIT_COMMON_ARG_H_
#define OPENMIT_COMMON_ARG_H_

#include <memory>
#include <string>
#include <vector>
#include "dmlc/io.h"
#include "dmlc/logging.h"
#include "dmlc/config.h"

namespace mit {
/*! \brief define arguments save type */
typedef std::vector<std::pair<std::string, std::string>> KWArgs;

/*! \brief class for arguments parse. */
class ArgParser {
  public:
    /*! \brief default constructor */
    ArgParser() : args_("") { }
    
    /*! \brief destructor */
    ~ArgParser() { }

    /*! \brief read argument from config file */
    void ReadFile(const char* filename) {
      std::unique_ptr<dmlc::Stream> fs(
          dmlc::Stream::Create(filename, "r"));
      //dmlc::Stream * fs = dmlc::Stream::Create(filename, "r");
      char buf[1000];
      while (true) {
        size_t r = fs->Read(buf, 1000);
        args_.append(buf, r);
        if (!r) break;
      }
      CHECK(!args_.empty())
        << "failed to read from " << filename;
    }

    /**
     * \brief read all args from config file
     */
    void ReadArgs(int argc, char* argv[]) {
      for (int i = 0; i < argc; ++i) {
        args_.append(argv[i]);
        args_.append(" ");
      }
    }

    /*! \brief get arguments */
    KWArgs GetKWArgs() {
      std::istringstream ss(args_);
      dmlc::Config *conf = new dmlc::Config(ss);

      KWArgs kwargs;
      for (const auto & entry : *conf) {
        kwargs.push_back(entry);
      }
      delete conf;
      return kwargs;
    }

    /*! \brief get arguments */
    std::string GetArgs() {
      return args_;
    }

  private:
    /*! \brief parsed arguments */
    std::string args_;
}; // class Argparser
} // namespace mit
#endif // OPENMIT_COMMON_ARG_H_
