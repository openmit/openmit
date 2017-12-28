/*!
 *  Copyright 2016 by Contributors
 *  \file optimizer.h
 *  \brief optimization algorithm
 *  \author ZhouYong
 */
#ifndef OPENMIT_OPTIMIZER_OPTIMIZER_H_
#define OPENMIT_OPTIMIZER_OPTIMIZER_H_

#include "ps/ps.h"
#include "clapack/blaswrap.h"
#include "clapack/f2c.h"
#include "clapack/clapack.h"
#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/parameter.h"
#include "openmit/entry/entry.h"
#include "openmit/tools/dstruct/sarray.h"



namespace mit {
/*!
 * \brief optimizer template for varies optimization algorithm
 */
class Optimizer {
  public:
    /*! \brief create a optimizer */
    static Optimizer * Create(const mit::KWArgs & kwargs, 
                              const std::string & name = "gd");
    
    /*! \brief destructor */
    virtual ~Optimizer() {}
    
    /*! 
     * \brief initialize optimizer middle variable
     * \param dim feature max dimension
     */
    virtual void Init(mit_uint dim) = 0;

    /*! \brief parameter updater for mpi */
    void Run(mit::SArray<mit_float> & grad, 
             mit::SArray<mit_float> * weight);

    /*! \brief parameter updater for ps */
    void Run(const ps::SArray<mit_uint> & keys, 
             const ps::SArray<mit_float> & vals, 
             const ps::SArray<int> & lens, 
             std::unordered_map<mit_uint, mit::Entry *> * weight);
  
    virtual void Update(const mit_uint & key, 
                        const size_t & idx, 
                        const mit_float & g, 
                        mit_float & w, 
                        mit::Entry * weight = nullptr) = 0;
    /*! 
     * \brief model updater for parameter server interface for als
     * \param user_keys user_id vector
     * \param user_weights user weights vector
     * \param user_lens user weights vector length
     * \param item_keys item_id vector
     * \param item_weights item weights vector
     * \param item_lens item weights vector length
     * \param user_res_vector user laten vector solved by als
     * \param item_res_vector item laten vector solved by als
     */
    virtual void Update(std::unordered_map<ps::Key, mit::mit_float>& rating_map,
                        std::vector<ps::Key>& user_keys,
                        std::vector<mit_float> & user_weights,
                        std::vector<int> & user_lens,
                        std::vector<ps::Key> & item_keys,
                        std::vector<mit_float> & item_weights,
                        std::vector<int> & item_lens,
                        std::vector<mit_float> * user_res_vector,
                        std::vector<mit_float> * item_res_vector) {}


  protected:
    /*! 
     * \brief model updater for parameter server interface
     * \param param optimizer parameter
     * \param key model feature id
     * \param idx entry data index
     * \param g gradient of unit index that computed by worker node
     * \param w model parameter of unit index 
     * \param weight used initialize optimizer middle variable
     */
    virtual void Update(const mit::OptimizerParam & param, 
                        const mit_uint & key, 
                        const size_t & idx, 
                        const mit_float & g,
                        mit_float & w,
                        mit::Entry * weight = nullptr) = 0;

    /*!
     * \brief parameter updater for mpi
     * \param idx model index 
     * \param g gradient of model index 
     * \param w model index weight
     */
    virtual void Update(const mit_uint idx, 
                        const mit_float g, 
                        mit_float & w) = 0;

   protected:
    /*! \brief optimizer parameter */
    mit::OptimizerParam param_;
    /*! \brief optimizer parameter for w */
    mit::OptimizerParam param_w_;
    /*! \brief optimizer parameter for v */
    mit::OptimizerParam param_v_;
    /*! \brief client parameter */
    mit::CliParam cli_param_;
}; // class Optimizer

} // namespace mit
#endif // OPENMIT_OPTIMIZER_OPTIMIZER_H_
