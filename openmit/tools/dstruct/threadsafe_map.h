/*!
 *  Copyright (c) 2016 by Contributors
 *  \file threadsafe_map.h
 *  \brief thread safe map structure 
 *  \author ZhouYong
 */
#ifndef OPENMIT_TOOLS_DSTRUCT_THREADSAFE_MAP_H_
#define OPENMIT_TOOLS_DSTRUCT_THREADSAFE_MAP_H_

#include <mutex>
#include <unordered_map>

namespace mit {
template <typename K, typename V>
class ThreadsafeMap {
  public:
    ThreadsafeMap() {}
    ~ThreadsafeMap() {}
    
    /*! \brief insert (thread safe) */
    void insert(K key, V& value) {
      std::lock_guard<std::mutex> lk(mu_);
      if (map_.find(key) != map_.end()) return;
      map_.insert(std::make_pair(key, value));
      auto* result = map_[key];
      printf("insert k: %d, v: %d, map_[%d]: %d\n", key, *value, key, *result);
    }
    
    /*! \brief find using key */
    typename std::map<K,V>::const_iterator find(K key) { return map_.find(key); }

    /*! \brief end op */
  typename std::map<K,V>::const_iterator end() { return map_.end(); }
    
    inline V operator[](K key) { 
      printf("[] key: %d, map_[%d]: %d\n", key, key, *map_[key]);
      return map_[key]; 
    }

    inline size_t size() const {
      return map_.size();
    }

  private:
  std::map<K, V*> map_;
    mutable std::mutex mu_;
};

}
#endif // OPENMIT_TOOLS_DSTRUCT_THREADSAFE_MAP_H_
