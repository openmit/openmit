#include "openmit/tools/monitor/transaction.h"
#include <iostream>
#include <unistd.h>
using namespace std;
using namespace mit; 

void Inner() {
  Transaction * trans = new Transaction(3, "gradient", "inner");
  Transaction::Create(trans);
  std::cout << "inner .... size: " << Transaction::Size() << std::endl;
  sleep(1);
  Transaction::End(trans);
  delete trans;
}

void Test() {
  Transaction * trans = Transaction::Create(1, "worker", "gradient");
  std::cout << "Test ..." << std::endl;
  for (int i = 0; i < 10; ++i) {
    Inner();
  }
  Transaction::End(trans); delete trans;
}

int main(int argc, char * argv[]) {
  mit::Transaction trans(0, "worker", "predict");
  std::cout << "type: " << trans.Type() << std::endl;
  Test();
  return 0;
}
