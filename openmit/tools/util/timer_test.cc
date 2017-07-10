#include <stdio.h>
#include "timer.h"

int main(int argc, char * argv[]) {
  double timer = mit::GetTime();
  unsigned long timestamp = mit::TimeStamp();
  printf("timer: %f, timestampe: %ld\n", timer, timestamp);
  return 0;
}
