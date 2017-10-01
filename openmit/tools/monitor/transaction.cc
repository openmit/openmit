#include "openmit/tools/monitor/transaction.h"

namespace mit {
// register static member
std::stack<Transaction *> Transaction::trans_info;
bool Transaction::_init = Transaction::Init();

bool Transaction::Init() {
  trans_info.push(new Transaction()); 
  trans_info.pop();
  return true;
}

Transaction * Transaction::Create(uint32_t level, 
                                  std::string type, 
                                  std::string name) {
  Transaction * trans = new Transaction(level, type, name);
  LOG(INFO) << "created transaction: <" << level << ", " 
    << type << ", " 
    << name << ">";
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
  delete trans_end;
}

void Transaction::LogTrace(Transaction * trans) {
  LOG(INFO) << "{\"level\":" << trans->Level() 
    << ", \"type\":" << trans->Type() 
    << ", \"name\":" << trans->Name() 
    << ", \"begin timestamp\":" << this->TimeStamp() 
    << ", \"end timestamp\":" << trans->TimeStamp() 
    << ", \"time consuming\":" << trans->TimeStamp() - this->TimeStamp()
    << "s}\n";
}
} // namespace mit
