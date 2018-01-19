#include "file.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
  std::string path("openmit");
  std::string cmd = "rm -rf " + path + " || true";
  std::cout << "cmd: " << cmd << std::endl;
  int rt = system(cmd.c_str());
  if (rt == 0) {
    std::cout << "system success" << std::endl;
  }
  
  if (mit::DirFile::access_dir(path.c_str())) {
    std::cout << path << " dir exists. rm it." << std::endl;
    if (mit::DirFile::rm_dir(path.c_str())) {
      std::cout << path << " rm success." << std::endl;
    } else {
      std::cout << path << " rm failure." << std::endl;
    }
  } else {
    std::cout << path << " dir not exists." << std::endl;
    if (mit::DirFile::mk_dir(path.c_str())) {
      std::cout << path << " create success." << std::endl;
    } else {
      std::cout << path << " create failure." << std::endl;
    }
  }
  return 0;
}
