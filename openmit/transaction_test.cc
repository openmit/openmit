/*
 * File Name: transaction_test.cc
 * Author: zhouyong03@meituan.com
 * Created Time: 2017-07-08 22:08:52
 */
 
#include "openmit/tools/monitor/transaction.h"
#include <iostream>
#include <unistd.h>
using namespace std;
using namespace mit; 

void Test() {
  Transaction * trans = new Transaction(1, "worker", "gradient");
  Transaction::Create(trans);
  std::cout << "Test ..." << std::endl;
  Transaction::End(trans);
}

int main(int argc, char * argv[]) {
  mit::Transaction trans(0, "worker", "predict");
  std::cout << "type: " << trans.Type() << std::endl;
  Test();
  return 0;
}
