#include "threadsafe_map.h"
#include <vector>

class Test {
  public:
    Test(int x, int y) : x_(x), y_(y) {}
    void Print() { printf("x_: %d, y_: %d\n",x_, y_ ); }
  private:
    int x_;
    int y_;
};

int main(int argc, char** argv) {
  mit::ThreadsafeMap<int, Test*> map;
  /*
  int M = 20;
  for (auto i = 0; i < M; ++i) {
    Test* test = new Test(i, i*2);
    map.insert(i, test);
    test = new Test(i, i*4);
    map.insert(i, test);
  }

  for (auto i = 0; i < M; ++i) {
    printf("i: %d, ", i);
    map[i]->Print();
  }
  */
  Test* res = map[10];
  if (res == NULL) {
    printf("1. res is NULL\n");
  } else {
    printf("1. res is not null\n");
  }
  res = map[10];
  if (res == NULL) {
    printf("1.5. res is null\n");
  } else {
    printf("1.5. res is not null\n");
  }
  if (map.find(10) == map.end()) {
    printf("2. res not exist\n");
  } else {
    printf("2. res exist\n");
  }
  return 0;
}
