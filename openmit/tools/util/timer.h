/*!
 *  Copyright 2016 by Contributors
 * \file timer.h
 * \brief timing
 * \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_UTIL_TIMER_H_
#define OPENMIT_TOOLS_UTIL_TIMER_H_

#include <time.h>
#include <iostream>

namespace mit {
/*!
 * \brief return time in seconds
 */
inline double GetTime(void) {
  timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
    return static_cast<double>(ts.tv_sec) + 
      static_cast<double>(ts.tv_nsec) * 1e-9;
  } else {
    return static_cast<double>(time(NULL));
  }
}
/*! 
 * \brief return timestamp
 */
inline unsigned long TimeStamp(void) {
  return static_cast<unsigned long>(time(NULL));
} 
} // namespace mit

#endif // OPENMIT_TOOLS_UTIL_TIMER_H_
