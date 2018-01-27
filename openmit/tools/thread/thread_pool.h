#ifndef OPENMIT_TOOLS_THREAD_THREAD_POOL_H_
#define OPENMIT_TOOLS_THREAD_THREAD_POOL_H_ 

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mit {
/*!
 * \brief a general thread pool, include return result type
 */
class ThreadPool {
  public:
    /*! \brief constructor and initialize thread pool */
    ThreadPool(size_t threads);
  
    /*! \brief destructor free thread */
    ~ThreadPool();

    /*! 
     * \brief append a task event
     * \param class F function 
     * \param Args arguments
     */
    template<class F, class... Args>
    auto Append(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

  private:
    /*! \brief worker threads */
    std::vector<std::thread> worker_threads_;
    /*! \brief ready task */
    std::queue<std::function<void()>> ready_tasks_;
    /*! \brief sync task */
    std::mutex mu_;
    std::condition_variable cond_;
    std::atomic<bool> stop_;
    
}; // class ThreadPool

inline ThreadPool::ThreadPool(size_t threads) : stop_(false){
  for (size_t i = 0; i < threads; ++i) {
    worker_threads_.emplace_back([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lk(this->mu_);
          this->cond_.wait(lk, [this] { return this->stop_ || !this->ready_tasks_.empty(); });
          if (this->stop_ && this->ready_tasks_.empty()) return;
          task = std::move(this->ready_tasks_.front());
          this->ready_tasks_.pop();
        }
        task();
      }
    });
  }
} // ThreadPool::ThreadPool 

inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lk(mu_);
    stop_ = true;
  }
  cond_.notify_all();
  for (auto&& thread : worker_threads_) {
    thread.join();
  }
} // ThreadPool::~ThreadPool

template<class F, class... Args>
auto ThreadPool::Append(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;
  
  auto task = std::make_shared<std::packaged_task<return_type()>>(
    std::bind(std::forward<F>(f), std::forward<Args>(args)...));
  
  std::future<return_type> rt = task->get_future();
  {
    std::unique_lock<std::mutex> lk(mu_);
    if (stop_) {
      throw std::runtime_error("ThreadPool has stopped. it can not Append op.");
    }
    ready_tasks_.emplace([task]() { (*task)(); });
  }
  cond_.notify_one();
  return rt;
} // ThreadPool::Append

} // namespace mit
#endif // OPENMIT_TOOLS_THREAD_THREAD_POOL_H_
