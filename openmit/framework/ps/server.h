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
#include "openmit/entity/entry.h"
#include "openmit/models/model.h"
#include "openmit/framework/ps/signal.h"

namespace mit {
/*!
 * \brief server processsor for distributed computate framework 
 */
class Server {
  public:
    /*! explicit constructor */
    explicit Server(const mit::KWArgs & kwargs);

    /*! virtual destructor */
    ~Server();
    
    /*! \brief initialize */
    void Init(const mit::KWArgs & kwargs);

    /*! \brief server core processing logic. */
    void Run(const ps::KVPairs<mit_float> & req_data);

  protected:
    /** 
     * \brief kv request handle logic 
     * \param req_meta request meta info
     * \param req_data request data info
     * \param server 
     */
    void KVRequestHandle(
        const ps::KVMeta & req_meta, 
        const ps::KVPairs<mit_float> & req_data,
        ps::KVServer<mit_float> * server);

    /*! 
     * \brief process pull request (weight)
     * \param req_data pull request information
     * \param response request response information
     */
    void ProcessPullRequest(const ps::KVPairs<mit_float> & req_data, 
                            ps::KVPairs<mit_float> & response);

  private:
    /*! \brief save model */
    void SaveModel(dmlc::Stream * fo);

    /*! \brief dump model */
    void DumpModel(dmlc::Stream * fi, dmlc::Stream * fo);
  
  private:
    /*! \brief client parameter info */
    mit::CliParam cli_param_;
    
    /*! \brief global model weight */
    std::unordered_map<ps::Key, mit::Entry * > weight_;
    
    /*! \brief process push & pull request */
    ps::KVServer<mit_float> * kv_server_;

    /*! \brief model */
    std::shared_ptr<mit::Model> model_;

    /*! \brief epoch complete worke number */
    //int epoch_complete_worker_num_;

    /*! \brief finalize after all worker done */
    //int complete_worker_num_;

    //DISALLOW_COPY_AND_ASSIGN(Server);

}; // class Server

} // namespace mit

#endif // OPENMIT_FRAMEWORK_PS_SERVER_H_
