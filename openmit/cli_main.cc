#include "openmit/common/arg.h"
#include "openmit/learner.h"
#include "clapack/blaswrap.h"
#include "clapack/f2c.h"
#include "clapack/clapack.h"

int main(int argc, char * argv[]) {
  CHECK_GE(argc, 2) << "Usage: " 
    << argv[0] << " openmit.conf [k1=v1] [k2=v2] ...";

  mit::ArgParser parser;
  if (strcmp(argv[1], "none")) parser.ReadFile(argv[1]);
  parser.ReadArgs(argc - 2, argv + 2);
  const mit::KWArgs kwargs = parser.GetKWArgs();

  mit::MILearner* mi = mit::MILearner::Create(kwargs);
  mi->Run();
  delete mi; mi = nullptr;
  return 0;
}

