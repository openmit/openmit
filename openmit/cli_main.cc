#include "openmit/common/arg.h"
#include "openmit/learner/mi_learner.h"
#include "openmit/tools/monitor/transaction.h"

int main(int argc, char * argv[]) {
  CHECK_GE(argc, 2)
    << "Usage: " << argv[0] << " mit.conf [k1=v1] [k2=v2] ...";

  mit::ArgParser parser;
  if (strcmp(argv[1], "none")) parser.ReadFile(argv[1]);
  parser.ReadArgs(argc - 2, argv + 2);
  const mit::KWArgs kwargs = parser.GetKWArgs();

  mit::Transaction * trans = mit::Transaction::Create(0, "ml-task", "milearner");
  auto * milearner = mit::MILearner::Create(kwargs);
  milearner->Run();
  mit::Transaction::End(trans);

  return 0;
}
