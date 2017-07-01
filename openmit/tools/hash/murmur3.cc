 /*!
 *  Copyright 2017 by Contributors
 *  \file murmur3.cc
 *  \brief hash algorithms : murmurhash3
 *  \author ZhouYong
 */

#include "murmur3.h"

#define CONSTANT(x) (x##LU)
#define BIG_CONSTANT(x) (x##LLU)

namespace mit {
namespace hash {

inline void rotl32(uint32_t *x, int8_t r) {
  *x = (*x << r) | (*x >> (32 - r));
}

inline void rotl64(uint64_t *x, int8_t r) {
  *x = (*x << r) | (*x >> (64 - r));
}

#define ROTL32(x,y) rotl32(x,y)
#define ROTL64(x,y) rotl64(x,y)

inline uint64_t getblock64(const uint64_t * p, int i) {
  return p[i];
}

void MMHash128::hash128(const void * key, const int len, 
    void * hash1, void * hash2, const uint32_t seed) {
  // ------------- initialization -----------  
  uint64_t h1 = seed, h2 = seed, k1, k2;
  const uint8_t * data = (const uint8_t *) key;
  const int n = len / 16;

  // ------------- constants ----------------
  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  const uint32_t c3 = CONSTANT(0x52dce729);
  const uint32_t c4 = CONSTANT(0x38495ab5);

  const uint64_t c5 = BIG_CONSTANT(0xff51afd7ed558ccd);
  const uint64_t c6 = BIG_CONSTANT(0xc4ceb9fe1a85ec53);

  // ------------- body ---------------------
  const uint64_t * blocks = (const uint64_t *) (data);

  for (int i = 0; i < n; i++) {
    k1 = getblock64(blocks, i*2+0);
    k1 = getblock64(blocks, i*2+1);

    k1 *= c1; ROTL64(&k1, 31); k1 *= c2; 
    h1 ^= k1; ROTL64(&h1, 27); h1 += h2; h1 = h1*5 + c3;
    k2 *= c2; ROTL64(&k2, 33); k2 *= c1;
    h2 ^= k2; ROTL64(&h2, 31); h2 += h1; h2 = h1*5 + c4;
  }

  // ------------- tail ---------------------
  k1 = 0; k2 = 0;
  const uint8_t * tail = (const uint8_t *) (data + n*16);
  switch (len & 15) {
    case 15: k2 ^= uint64_t(tail[14]) << 48;
    case 14: k2 ^= uint64_t(tail[13]) << 40;
    case 13: k2 ^= uint64_t(tail[12]) << 32;
    case 12: k2 ^= uint64_t(tail[11]) << 24;
    case 11: k2 ^= uint64_t(tail[10]) << 16;
    case 10: k2 ^= uint64_t(tail[ 9]) << 8;
    case  9: k2 ^= uint64_t(tail[ 8]) << 0;
             k2 *= c2; ROTL64(&k2, 33); k2 *= c1; h2 ^= k2;
    case  8: k1 ^= uint64_t(tail[ 7]) << 56;
    case  7: k1 ^= uint64_t(tail[ 6]) << 48;
    case  6: k1 ^= uint64_t(tail[ 5]) << 40;
    case  5: k1 ^= uint64_t(tail[ 4]) << 32;
    case  4: k1 ^= uint64_t(tail[ 3]) << 24;
    case  3: k1 ^= uint64_t(tail[ 2]) << 16;
    case  2: k1 ^= uint64_t(tail[ 1]) << 8;
    case  1: k1 ^= uint64_t(tail[ 0]) << 0;
             k1 *= c1; ROTL64(&k1, 31); k1 *= c2; h1 ^= k1;
  };

  // ------------- finalization ----------------
  h1 ^= len; h2 ^= len; 
  h1 += h2; h2 += h1;

  h1 ^= h1 >> 33; h1 *= c5; h1 ^= h1 >> 33; h1 *= c6; h1 ^= h1 >> 33;
  h2 ^= h2 >> 33; h2 *= c5; h2 ^= h2 >> 33; h2 *= c6; h2 ^= h2 >> 33;
  
  h1 += h2; h2 += h1;

  // ------------- output ---------------------
  *(uint64_t *)hash1 = h1;
  *(uint64_t *)hash2 = h2;
}

void MMHash128::hash128(
    void const * key, 
    const int size, 
    void * hash, 
    uint32_t seed) {
  uint64_t *h1 = reinterpret_cast<uint64_t *>(hash);
  uint64_t *h2 = h1 + 1;
  return hash128(key, size, h1, h2, seed);
}

void MMHash32::hash32(
    const void * key,
    const int len,
    void * hash,
    const uint32_t seed) {
  // ------------- initialization -------------
  uint32_t h = seed, k;

  // ------------- constant -------------------
  const uint32_t c1 = CONSTANT(0xcc9e2d51);
  const uint32_t c2 = CONSTANT(0x1b873593);
  const uint32_t c3 = CONSTANT(0xe6546b64);
  const uint32_t c4 = CONSTANT(0x85ebca6b);
  const uint32_t c5 = CONSTANT(0xc2b2ae35);
  
  // ------------- body ---------------------
  const uint32_t * data = (const uint32_t *) key;
  const int n = len / 4;
  for (int i = 0; i < n; ++i) {
    k = *data++;
    k *= c1; ROTL32(&k, 15); k *= c2;
    h ^= k; ROTL32(&h, 13); h = h*5 + c3;
  }
  
  // ------------- tail ---------------------
  k = 0;
  const uint8_t * tail = (const uint8_t *) (data);
  switch (len & 3) {
    case 3: k ^= uint32_t(tail[2]) << 16;
    case 2: k ^= uint32_t(tail[1]) << 8;
    case 1: k ^= uint32_t(tail[0]);
            k *= c1; ROTL32(&k, 15); k *= c2; h ^= k;
  };
  
  // ------------- finalization ----------------
  h ^= len;
  h ^= h >> 16; h *= c4; h ^= h >> 13; h *= c5; h ^= h >> 16;
  
  // ------------- tail ---------------------
  *(uint32_t *)hash = h;
}

uint32_t MMHash32::hash32(const void * key, 
                          const int len, 
                          const uint32_t seed) {
  uint32_t hash;
  hash32(key, len, &hash, seed);
  return hash;
}

} // namespace hash
} // namespace mit
