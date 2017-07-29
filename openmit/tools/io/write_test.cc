/*
 * File Name: rtest.cc
 * Author: zhouyongsdzh@foxmail.com
 * Created Time: 2017-07-29 15:29:07
 */

#include <fstream>
#include <iostream>
#include "openmit/tools/io/write.h"
using namespace std;
 
void write(ofstream & ofs, const char * content) {
  ofs << content << endl;
}

int main(int argc, char ** argv) {
  std::string str = "I love this world!!!!";
  mit::Write write(argv[1]);
  write.write_line(str.c_str());
  write.write_line(str.c_str());
  write.write(str.c_str());
  write.write_line(str.c_str());
  write.write_line(str.c_str());
  write.write_line(str.c_str());
  /*
  ofstream ofs(argv[1], std::ios_base::trunc);
  write(ofs, str.c_str());
  write(ofs, str.c_str());
  write(ofs, str.c_str());
  write(ofs, str.c_str());
  write(ofs, str.c_str());
  ofs << str.c_str() << endl;
  ofs << str.c_str() << endl;
  ofs << str.c_str() << endl;
  ofs << str.c_str() << endl;
  ofs.close();
  */
  return 0;
}
