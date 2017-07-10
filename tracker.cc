#include "openmit/tools/monitor/tracker.h"
#include <iostream>

namespace mit {
// register parameter
DMLC_REGISTER_PARAMETER(TrackerParam);

void Tracker::Init(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
}

Tracker * Tracker::Create(uint32_t level, 
                              std::string type,
                              std::string name) {
  Transaction * trans = new Transaction(level, type, name);
  trace_info.push(trans);
  return trans;
}

void Tracker::End(Transaction * trans) {
  CHECK(!trace_info.empty()) 
    << "stack trace_info empty! no element to pop!";
  Transaction * trans_begin = trace_info.top();
  Transaction * trans_end = new Transaction(
      trans_begin->Level(), 
      trans_begin->Type(), 
      trans_begin->Name());
  if (param_.is_trace && 
      trans_begin->Level() <= param_.trace_level) {
    LogTrace(trans_begin, trans_end);
  }
  trace_info.pop();
  delete trans_end;
  delete trans_begin;
}

void Tracker::LogTrace(Transaction * begin, Transaction * end) {
  std::cout << "{\"level\":" << begin->Level() 
    << ",\"type\":" << begin->Type() 
    << ",\"name\":" << begin->Name() 
    <<  ",\"timestampe1\":" << begin->TimeStamp() 
    << "\"timestamp2\":" << end->TimeStamp() 
    << "\"time consuming\":" << (end->TimeStamp()-begin->TimeStamp())/60
    << "min}" << std::endl;
}

} // namespace mit
