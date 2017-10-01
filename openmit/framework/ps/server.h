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

#include "ps/ps.h"
#include "ps/sarray.h"
#include "dmlc/io.h"
#include "dmlc/parameter.h"

#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/engine/updater.h"
#include "openmit/entity/unit.h"
#include "openmit/framework/ps/signal.h"

namespace mit {
/*!
 * \brief server related parameter
 */
class ServerParam : public dmlc::Parameter<ServerParam> {
  public:
    /*! \brief task type. */
    std::string task_type;
    /*! \brief model type. "lr", "fm", "ffm", "mf", ... */
    std::string model;
    /*! \brief optimizer type. "sgd", "adagrad", "ftrl", "lbfgs", "als" */
    std::string optimizer;
    /*! \brief sync mode. "asp", "bsp", "ssp" */
    std::string sync_mode;
    /*! \brief model input path */
    std::string model_in;
    /*! \brief model output path */
    std::string model_dump;
    /*! \brief model binary path */
    std::string model_binary;
    /*! \brief field number */
    size_t field_num;
    /*! \brief latent vector length for fm/ffm. default=1 */
    size_t embedding_size;

    /*! \brief declare parameters */
    DMLC_DECLARE_PARAMETER(ServerParam) {
      DMLC_DECLARE_FIELD(task_type).set_default("train");
      DMLC_DECLARE_FIELD(model).set_default("lr");
      DMLC_DECLARE_FIELD(optimizer).set_default("sgd");
      DMLC_DECLARE_FIELD(sync_mode).set_default("sync");
      DMLC_DECLARE_FIELD(model_in).set_default("");
      DMLC_DECLARE_FIELD(model_dump).set_default("");
      DMLC_DECLARE_FIELD(model_binary).set_default("");
      DMLC_DECLARE_FIELD(field_num).set_default(10);
      DMLC_DECLARE_FIELD(embedding_size).set_default(4);
    }
}; // ServerParam

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
    void Run(const ps::KVPairs<mit_float> * req_data);

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

  private:
    /*! \brief save model */
    void SaveModel(dmlc::Stream * fo);

    /*! \brief dump model */
    void DumpModel(dmlc::Stream * fi, dmlc::Stream * fo);
  
  private:
    /*! \brief server parameter info */
    mit::ServerParam param_;
    
    /*! \brief process push & pull request */
    ps::KVServer<mit_float> * kv_server_;
    
    /*! \brief updater */
    std::shared_ptr<mit::Updater> updater_;

    /*! \brief global model weight */
    std::unordered_map<ps::Key, mit::Unit * > weight_;

    /*! \brief mutex */
    //std::mutex mu_;
    
    /*! \brief condition */
    //std::condition_variable cond_;

    /*! \brief mutex used exit */
    //std::mutex exit_mu_;

    /*! \brief eixt when scheduler send message to server */
    //bool exit_ = false;

    /*! \brief whether doing sync */
    //bool doing_sync_;

    /*!
     * \brief save gradient from workers for sync model update
     */
    //std::vector<std::unordered_map<mit_uint, std::vector<mit_float> > > grad_;

    /*! \brief epoch complete worke number */
    //int epoch_complete_worker_num_;

    /*! \brief finalize after all worker done */
    //int complete_worker_num_;

    //DISALLOW_COPY_AND_ASSIGN(Server);

}; // class Server

} // namespace mit

#endif // OPENMIT_FRAMEWORK_PS_SERVER_H_
