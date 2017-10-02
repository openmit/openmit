#include "openmit/tools/monitor/transaction.h"

namespace mit {
// register static member
std::stack<Transaction *> Transaction::trans_info;
bool Transaction::_init = Transaction::Init();

bool Transaction::Init() {
  /*
  trans_info.push(new Transaction(99, "type", "name")); 
  if (! trans_info.empty()) trans_info.pop();
  // TODO error reported by valgrind 
  ==5294== 88 bytes in 1 blocks are definitely lost in loss record 23 of 30
  ==5294==    at 0x4C2E118: operator new(unsigned long) (vg_replace_malloc.c:333)
  ==5294==    by 0x5314B9: mit::Transaction::Init() (in ...)
  ==5294==    by 0x531B79: __static_initialization_and_destruction_0(int, int) (in ...)
  */
  return true;
}

Transaction * Transaction::Create(uint32_t level, 
                                  std::string type, 
                                  std::string name) {
  Transaction * trans = new Transaction(level, type, name);
  LOG(INFO) << "created transaction: <" 
    << level << ", " << type << ", " << name << ">";
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
  if (! trans_info.empty()) trans_info.pop();
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
