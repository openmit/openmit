#include "openmit/tools/monitor/transaction.h"

namespace mit {
// register static member
std::stack<Transaction *> Transaction::trans_info;
bool Transaction::_init = Transaction::Init();

bool Transaction::Init() {
  trans_info.push(new Transaction());
  return true;
}

Transaction * Transaction::Create(uint32_t level, 
                                  std::string type, 
                                  std::string name) {
  Transaction * trans = new Transaction(level, type, name);
  std::cout << "transaction: <" << level << ", " 
    << type << ", " 
    << name << "> create" << std::endl;
  trans_info.push(trans);
  return trans;
}

void Transaction::Create(Transaction * trans) {
  trans_info.push(trans);
}

void Transaction::End(Transaction * trans) {
  Transaction * trans_end = new Transaction(
      trans->Level(), trans->Type(), trans->Name());
  trans->LogTrace(trans_end);
  trans_info.pop();
  delete trans;
  delete trans_end;
}

void Transaction::LogTrace(Transaction * trans) {
  std::cout << "{\"level\":" << trans->Level() 
    << ",\"type\":" << trans->Type() 
    << ",\"name\":" << trans->Name() 
    <<  ",\"timestampe1\":" << this->TimeStamp() 
    << ",\"timestamp2\":" << trans->TimeStamp() 
    << ",\"time consuming\":" << trans->TimeStamp() - this->TimeStamp()
    << "s}" << std::endl;
}
} // namespace mit
