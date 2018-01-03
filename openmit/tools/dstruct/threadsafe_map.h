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
#include <unistd.h>

namespace mit {
template <typename K, typename V>
class ThreadsafeMap {
  public:
    ThreadsafeMap() {}
    ~ThreadsafeMap() {}
  
    /*! \brief insert (thread safe) */
    void insert(std::pair<K,V> kv) {
      CHECK_NOTNULL(kv.second);
      if (map_.find(kv.first) != map_.end()) {
        //LOG(INFO) << kv.first << "-has exist in threadsafe_map";
        return;
      }
      std::lock_guard<std::mutex> lk(mutex_write_);
      map_.insert(kv);
      if (map_.find(kv.first) == map_.end() || map_[kv.first] == nullptr) {
        LOG(FATAL) << "map_[" << kv.first << "] error";
      }
    }

    /*! \brief insert (thread safe) */
    void insert(K key, V value) {
      if (map_.find(key) != map_.end()) {
        return;
      }
      std::lock_guard<std::mutex> lk(mutex_write_);
      //map_.insert(std::make_pair(std::move(key), std::move(value)));
      map_.insert(std::make_pair(key, value));
    }
    
    /*! \brief find using key */
    typename std::unordered_map<K,V>::iterator find(K key) { 
      std::lock_guard<std::mutex> lk(mutex_find_);
      return map_.find(key);
    }

    /*! \brief begin iterator */
    typename std::unordered_map<K,V>::iterator begin() { return map_.begin(); }

    /*! \brief end iterator */
    typename std::unordered_map<K,V>::iterator end() { return map_.end(); }
    
    inline V operator[](K key) {
      std::lock_guard<std::mutex> lk(mutex_read_);
      if (find(key) == end()) {
        LOG(INFO) << key << " not exist. sleep 1s aaaaaaaa";
        sleep(1);
        if (find(key) == end()) {
          LOG(FATAL) << key << " not exist fatal. bbbbbbb";
          return 0;
        }
      }
      return map_[key];
    }

    inline size_t size() const { return map_.size(); }

  private:
    std::unordered_map<K, V> map_;
    std::mutex mutex_write_;
    std::mutex mutex_read_;
    std::mutex mutex_find_;
};

}
#endif // OPENMIT_TOOLS_DSTRUCT_THREADSAFE_MAP_H_
