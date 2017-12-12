#include <unistd.h>
#include <vector>
#include <pmmintrin.h>
#include "openmit/tools/util/timer.h"
#include <iostream>

int embedding_size_ = 4;

int field_offset(const std::pair<uint32_t, const float *> & wv, const uint32_t & fieldid) {
  if (wv.first <= 1) return -1;
  for (uint32_t i = 0; i < wv.first; i += (1 + embedding_size_)) {
    if (fieldid == static_cast<uint32_t>(*(wv.second + i))) {
      return i;
    }
  }
  return -1;
}


// version 1
float sum1(const float * val, unsigned int size) {
  float sum = 0;
  for (unsigned int i = 0; i < size; ++i) {
    sum += val[i];
  }
  return sum;
}

// version 2
float sum2(const float * val, unsigned int size) {
  float sum = 0;
  int nBlockWidth = 4;  // SSE
  int cntBlock = size / nBlockWidth;
  int cntMem = size % nBlockWidth;
  __m128 fSum = _mm_setzero_ps();
  __m128 fLoad;

  const float * p = val;
  for (int i = 0; i < cntBlock; ++i) {
    fLoad = _mm_load_ps(p);
    fSum = _mm_add_ps(fSum, fLoad);
    p += nBlockWidth;
  }
  const float *q = (const float *)&fSum;
  sum = q[0] + q[1] + q[2] + q[3];
  for (int i = 0; i < cntMem; ++i) {
    sum += p[i];
  }
  return sum;
}


int main(int argc, char ** argv) {
  float op1[4] = {1.0, 2.0, 3.0, 4.0};
  float op2[4] = {1.0, 2.0, 3.0, 4.0};
  float result[4];

  __m128 a1; 
  __m128 a2;
  __m128 c;

  // load
  a1 = _mm_loadu_ps(op1);
  a2 = _mm_loadu_ps(op2);

  // calculate 
  c = _mm_add_ps(a1, a2);  // c = a1 + a2;

  // store 
  _mm_storeu_ps(result, c);

  printf("%d: %lf\n", 0, result[0]);
  printf("%d: %lf\n", 1, result[1]);
  printf("%d: %lf\n", 2, result[2]);
  printf("%d: %lf\n", 3, result[3]);

  long count = std::atol(argv[1]);
  printf("count: %ld\n",count);
  /*
  std::vector<float> test1;
  for (int i = 0; i < 100; ++i) {
    test1.push_back(i);
  }
  double start_time1 = mit::GetTime();
  printf("start_time: %f\n", start_time1);
  int offset;
  for (long i = 0; i < count; ++i) {
    offset = field_offset(std::pair<uint32_t, const float *>(100, test1.data()), 80);
  }
  printf("offset: %d\n", offset);
  double end_time1 = mit::GetTime();
  printf("end_time: %f\n", end_time1);
  printf("time consume: %f s\n", (end_time1 - start_time1));
  */
  std::vector<float> a;
  a.push_back(1);
  a.push_back(2);
  a.push_back(3);
  a.push_back(4);
  std::vector<float> b;
  b.push_back(1);
  b.push_back(2);
  b.push_back(3);
  b.push_back(4);
  int embedding_size = 4;
    const float * pa = a.data();
    const float * pb = b.data();
  double start_time = mit::GetTime();
  printf("start_time: %f\n", start_time);
  __m128 tmp = _mm_setzero_ps();
  float wTx = 0;
  for (long i = 0; i < count; ++i) {
    /*
    wTx = 0;
    for (int f = 0; f < embedding_size; f++) {
      wTx += pa[f] * pb[f];
    }
    */
    tmp = _mm_setzero_ps();
    for (int f = 0; f < embedding_size; f += 4) {
      __m128 v1 = _mm_load_ps(pa+f);
      __m128 v2 = _mm_load_ps(pb+f);
      tmp = _mm_add_ps(tmp,  _mm_mul_ps(v1, v2));
    }
    const float *p = (const float *)&tmp;
    wTx = p[0] + p[1] + p[2] + p[3];
    //tmp = _mm_hadd_ps(tmp, tmp);
    //tmp = _mm_hadd_ps(tmp, tmp);
    //_mm_store_ss(&wTx, tmp);
    //printf("i: %d, wTx: %f\n", i, wTx);
  }
  printf("wTx: %f\n", wTx);
  double end_time = mit::GetTime();
  printf("end_time: %f\n", end_time);
  printf("time consume: %f s\n", (end_time - start_time));
  return 0;
}
