#ifndef OPENMIT_COMMON_TYPE_H_
#define OPENMIT_COMMON_TYPE_H_ 

#include "tbb/concurrent_unordered_map.h"
#include "openmit/entry/entry.h"

namespace mit {
/*! brief ps frameworker parameter type */
// model parameter type in server side 
typedef tbb::concurrent_unordered_map<long, Entry*> entry_map_type;

}
#endif // OPENMIT_COMMON_TYPE_H_
