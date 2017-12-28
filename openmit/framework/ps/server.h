/*!
 *  Copyright 2017 by Contributors
 *  \file server.h
 *  \brief server logic for parameter server
 *  \author ZhouYong
 */
#ifndef OPENMIT_FRAMEWORK_PS_SERVER_H_
#define OPENMIT_FRAMEWORK_PS_SERVER_H_

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
#include "openmit/entry/entry.h"
#include "openmit/model/psmodel.h"
#include "openmit/framework/ps/signal.h"

namespace mit {
/*!
 * \brief server processsor for distributed computation framework 
 */
class Server {
  public:
    /*! explicit constructor */
    explicit Server(const mit::KWArgs & kwargs);

    /*! virtual destructor */
    ~Server();
    
    /*! \brief initialize */
    void Init(const mit::KWArgs & kwargs);

    /*! \brief main process logic. */
    void Run();

  protected:
    /*! 
     * \brief kv request handle logic 
     * \param req_meta request meta info
     * \param req_data request data info
     * \param server 
     */
    void KVHandle(const ps::KVMeta & req_meta, 
                  const ps::KVPairs<mit_float> & req_data, 
                  ps::KVServer<mit_float> * server);

    /*!
     * \brief signal process handle logic
     */
    void CmdHandle(const ps::SimpleData & recved, 
                   ps::SimpleApp * app);

    /*! 
     * \brief process pull request (weight)
     * \param req_data pull request information
     * \param response request response information
     */
    void PullRequest(const ps::KVPairs<mit_float> & req_data, 
                     ps::KVPairs<mit_float> & response);

    /*! 
     * \brief logic for worker finish
     */
    void ExitCondition();

  private:
    /*! \brief save model */
    void SaveModel(std::string epoch = "", std::string prefix = "");

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
    ps::KVServer<mit_float> * kv_server_; 
    /*! \brief global model weight */
    std::unordered_map<ps::Key, mit::Entry *> weight_;
    /*! \brief model for pull request */
    std::shared_ptr<mit::PSModel> model_;
    /*! \brief updater used for parameter update */
    //std::unique_ptr<mit::Updater> updater_;

    /*! \brief mutex */
    std::mutex mutex_;

    /*! \brief control task exit condition */
    std::condition_variable cond_;

    /*! \brief task exit tag */
    bool exit_ = false;

    /*! \brief finalize after all worker done */
    int complete_worker_number_;

    //DISALLOW_COPY_AND_ASSIGN(Server);

}; // class Server

} // namespace mit

#endif // OPENMIT_FRAMEWORK_PS_SERVER_H_
