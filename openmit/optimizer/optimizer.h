/*!
 *  Copyright 2017 by Contributors
 *  \file optimizer.h
 *  \brief basic match formula, such as sigmoid, distance etc.
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
    /*! \brief create a optimization algorithm */
    static Opt * Create(
        const mit::KWArgs & kwargs, 
        std::string & optimizer);
    /*! \brief destructor */
    virtual ~Opt() {}
    /*! \brief parameter updater for mpi */
    virtual void Update(
        const dmlc::Row<mit_uint> & row, 
        mit_float pred, 
        mit::SArray<mit_float> & weight_) = 0;
    /*! \brief parameter updater for ps */
    virtual void Update(
        std::unordered_map<ps::Key, mit::Unit * > & map_grad,
        std::unordered_map<ps::Key, mit::Unit * > * weight) = 0;
}; // class Opt
} // namespace mit

#endif // OPENMIT_OPTIMIZER_OPTIMIZER_H_
