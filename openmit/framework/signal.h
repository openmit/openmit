#ifndef OPENMIT_FRAMEWORK_PS_SIGNAL_H_
#define OPENMIT_FRAMEWORK_PS_SIGNAL_H_

namespace mit {
namespace signal {
/*!
 * \brief signal instruction for scheduler/worker/server 
 *        three party interaction
 */
enum CommSignals {
  // signals: worker --> server
  UPDATE,
  SAVE_EPOCH,
  // signals: worker --> scheduler && server
  WORKER_FINISH,
  // signals: worker --> scheduler
  METRIC,
  LOSS_FUNC,
  // signals: server --> scheduler 
  SERVER_FINISH,
  // signals: scheduler --> worker/server 
  JOB_DONE,
  EARLY_STOP
};

} // namespace ps
} // namespace mit

#endif // OPENMIT_FRAMEWORK_PS_SIGNAL_H_
