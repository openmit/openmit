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
#include "openmit/tools/dstruct/sarray.h"

namespace mit {
/*!
 * \brief optimizer template for varies optimization algorithm
 */
class Opt {
  public:
    /*! \brief create a optimizer */
    static Opt * Create(const mit::KWArgs & kwargs, 
                        std::string & optimizer);
    
    /*! \brief destructor */
    virtual ~Opt() {}
    
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

    /*!
     * \brief parameter updater for mpi
     * \param idx model index 
     * \param g gradient of model index 
     * \param w model index weight
     */
    virtual void Update(const mit_uint idx, 
                        const mit_float g, 
                        mit_float & w) = 0;

}; // class Opt
} // namespace mit

#endif // OPENMIT_OPTIMIZER_OPTIMIZER_H_
