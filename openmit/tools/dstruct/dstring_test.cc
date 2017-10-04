#include "dstring.h"
#include <iostream>

int main(int argc, char ** argv) {
  std::string str("auc");
  std::vector<std::string> result;
  mit::string::Split(str, &result, ',');
  for (size_t i = 0; i < result.size(); ++i) {
    std::cout << result[i] << std::endl;
  }
  return 0;
}
