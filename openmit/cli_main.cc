#include <iostream>

#include "openmit/common/arg.h"
#include "openmit/tools/math/basic_formula.h"
#include "openmit/learner/mi_learner.h"

int main(int argc, char * argv[]) {
  CHECK_GE(argc, 2)
    << "Usage: " << argv[0] << "mit.conf [k1=v1] [k2=v2] ...";

  mit::ArgParser parser;
  if (strcmp(argv[1], "none")) parser.ReadFile(argv[1]);
  parser.ReadArgs(argc - 2, argv + 2);
  const mit::KWArgs kwargs = parser.GetKWArgs();

  auto * milearner = mit::MILearner::Create(kwargs);
  milearner->Run();

  return 0;
}
