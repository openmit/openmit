#include "openmit/tools/monitor/tracker.h"
using namespace mit;

void TrackerTest() {
  Transaction * trans = Tracker::Create(1, "worker", "gradient");
  std::cout << "TrackerTest ...." << std::endl;
  Tracker::End(trans);
}

int main(int argc, char * argv[]) {
  CHECK_GE(argc, 2)
    << "Usage: " << argv[0] << " mit.conf [k1=v1] [k2=v2] ...";

  mit::ArgParser parser;
  if (strcmp(argv[1], "none")) parser.ReadFile(argv[1]);
  parser.ReadArgs(argc - 2, argv + 2);
  const mit::KWArgs kwargs = parser.GetKWArgs();

  Tracker::Init(kwargs);

  TrackerTest();
  return 0;
}
