#include <iostream>
#include "type_conversion.h"

int main(int argc, char ** argv) {
  const char * xx = "123.456";
  float v = mit::StringToNum<float>(std::string(xx));
  std::cout << "v: " << v << "\n";
  return 0;  
}
