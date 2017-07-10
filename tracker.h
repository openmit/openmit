#ifndef OPENMIT_TOOLS_MONITOR_TRACKER_H_
#define OPENMIT_TOOLS_MONITOR_TRACKER_H_

#include <stack>
#include "dmlc/logging.h"
#include "dmlc/parameter.h"
#include "openmit/common/arg.h"
#include "openmit/tools/monitor/transaction.h"

namespace mit {
/*! 
 * \brief tracker parameter 
 */
class TrackerParam : public dmlc::Parameter<TrackerParam> {
  public:
    /*! \brief is trace */
    bool is_trace;
    /*! \brief trace level */
    uint32_t trace_level;

    // declare parameter field
    DMLC_DECLARE_PARAMETER(TrackerParam) {
      DMLC_DECLARE_FIELD(is_trace).set_default(true);
      DMLC_DECLARE_FIELD(trace_level).set_default(3);
    }
}; // class TrackerParam

/*! 
 * \brief tracker 
 */
class Tracker {
  public:
    Tracker();
    ~Tracker() {}

    static void Init(const mit::KWArgs & kwargs);

    static Tracker * Create(uint32_t level, 
                                std::string type, 
                                std::string name);
    /*! \brief */
    static void End(Tracker * tracker, 
                    bool is_trace = true, 
                    uint32_t trace_level = 3);


  private:
    /*! \brief */
    static void LogTrace(Transaction * begin, Transaction * end);

  private:
    /*! \brief trace info */
    static std::stack<Transaction *> trace_info;
}; // class Tracker

} // namespace mit

#endif // OPENMIT_TOOLS_MONITOR_TRACKER_H_
