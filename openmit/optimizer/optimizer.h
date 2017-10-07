/*!
 *  Copyright 2016 by Contributors
 *  \file optimizer.h
 *  \brief optimization algorithm
 *  \author ZhouYong
 */
#ifndef OPENMIT_OPTIMIZER_OPTIMIZER_H_
#define OPENMIT_OPTIMIZER_OPTIMIZER_H_

#include "ps/ps.h"

#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/data/data.h"
#include "openmit/entity/unit.h"
#include "openmit/entity/entry.h"
#include "openmit/tools/dstruct/sarray.h"

namespace mit {

struct OptimizerParam {
  /*! \brief learning rate */
  float lr = 0.01;
  // TODO all optimizer parameter
}; // class optimizer

/*!
 * \brief optimizer template for varies optimization algorithm
 */
class Optimizer {
  public:
    /*! \brief create a optimizer */
    static Optimizer * Create(const mit::KWArgs & kwargs, 
                        std::string & optimizer);
    
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
    void Run(PMAPT & map_grad, PMAPT * weight);
    
    void Run(const ps::SArray<mit_uint> & keys, 
             const ps::SArray<mit_float> & vals, 
             const ps::SArray<int> & lens, 
             PMAPT1 * weight);
  
  protected:
    /*! 
     * \brief model updater for parameter server interface
     * \param key model feature id
     * \param idx model unit index
     * \param size model unit max size
     * \param g gradient of unit index that computed by worker node
     * \param w model parameter of unit index
     */
    virtual void Update(const mit_uint key, 
                        const uint32_t idx, 
                        const uint32_t size, 
                        const mit_float g, 
                        mit_float & w) = 0;
    
    /*! \brief optimizer implementation for ps interface */
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
    mit::OptimizerParam param_w_;
    mit::OptimizerParam param_v_;
}; // class Optimizer

} // namespace mit
#endif // OPENMIT_OPTIMIZER_OPTIMIZER_H_
