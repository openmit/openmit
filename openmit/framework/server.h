/*!
 *  Copyright 2017 by Contributors
 *  \file server.h
 *  \brief server logic for parameter server
 *  \author ZhouYong
 */
#ifndef OPENMIT_FRAMEWORKgSERVER_H_
#define OPENMIT_FRAMEWORKgSERVER_H_

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "dmlc/io.h"
#include "dmlc/parameter.h"
#include "ps/ps.h"
#include "ps/sarray.h"

#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/type.h"
#include "openmit/entry/entry.h"
#include "openmit/model/model.h"
#include "openmit/framework/signal.h"
#include "openmit/tools/thread/thread_pool.h"

namespace mit {
/*!
 * \brief server processsor for distributed computation framework 
 */
class Server {
  public:
    /*! explicit constructor */
    explicit Server(const mit::KWArgs& kwargs);

    /*! virtual destructor */
    ~Server();
    
    /*! \brief initialize */
    void Init(const mit::KWArgs& kwargs);

    /*! \brief main process logic. */
    void Run();

  protected:
    /*! 
     * \brief kv request handle logic 
     * \param req_meta request meta info
     * \param req_data request data info
     * \param server 
     */
    void KVHandle(const ps::KVMeta& req_meta, 
                  const ps::KVPairs<mit_float>& req_data, 
                  ps::KVServer<mit_float>* server);

    /*!
     * \brief signal process handle logic
     */
    void CmdHandle(const ps::SimpleData& recved, 
                   ps::SimpleApp* app);

    /*! 
     * \brief process pull request (weight)
     * \param req_data pull request information
     * \param response request response information
     */
    void PullRequest(const ps::KVMeta& req_meta, 
                     const ps::KVPairs<mit_float>& req_data, 
                     ps::KVServer<mit_float>* server);

    /*! 
     * \brief logic for worker finish
     */
    void ExitCondition();

  private:
    /*! \brief save model */
    void SaveModel(std::string epoch = "");

    /*! \brief save text model */
    void SaveTextModel(dmlc::Stream * fo);

    /*! \brief save binary model */
    void SaveBinaryModel(dmlc::Stream * fo);

    /*! \brief dump model */
    void DumpModel(dmlc::Stream * fi, dmlc::Stream * fo);

    /*! \brief load model used to prediction */
    void LoadModel(dmlc::Stream * fi);
  
  private:
    /*! \brief client parameter info */
    mit::CliParam cli_param_;
    
    /*! \brief process kv request, such as pull/push */
    ps::KVServer<mit_float>* kv_server_; 
    
    /*! \brief global model weight */
    mit::entry_map_type weight_;
    
    /*! \brief model for server op: update && pull */
    std::shared_ptr<mit::Model> model_;

    /*! \brief thread pool for pull request */
    std::shared_ptr<mit::ThreadPool> thread_pool_;

    /*! \brief finalize after all worker done */
    int complete_worker_number_;

    /*! \brief sync variable */
    std::mutex mutex_;
    std::condition_variable cond_;
    bool exit_ = false;

    DISALLOW_COPY_AND_ASSIGN(Server);

}; // class Server

} // namespace mit

#endif // OPENMIT_FRAMEWORKgSERVER_H_
