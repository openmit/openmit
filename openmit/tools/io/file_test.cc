#include "file.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
  char* path = "openmit";
  std::string cmd = "rm -rf " + std::string(path) + " || true";
  std::cout << "cmd: " << cmd << std::endl;
  system(cmd.c_str());
  
  if (mit::DirFile::access_dir(path)) {
    std::cout << path << " dir exists. rm it." << std::endl;
    if (mit::DirFile::rm_dir(path)) {
      std::cout << path << " rm success." << std::endl;
    } else {
      std::cout << path << " rm failure." << std::endl;
    }
  } else {
    std::cout << path << " dir not exists." << std::endl;
    if (mit::DirFile::mk_dir(path)) {
      std::cout << path << " create success." << std::endl;
    } else {
      std::cout << path << " create failure." << std::endl;
    }
  }
  return 0;
}
