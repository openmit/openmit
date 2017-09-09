 /*!
 *  Copyright 2016 by Contributors
 *  \file adam.h
 *  \brief the Adam optimization algorithm
 *  \author ZhouYong
 */
#ifndef OPENMIT_OPTIMIZER_ADAM_H_
#define OPENMIT_OPTIMIZER_ADAM_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {
/*! 
 * \brief adam parameter
 */
class AdamParam : public dmlc::Parameter<AdamParam> {
  public:
    // TODO
}; // class Adam

/*!
 * \brief adam optimization algorithm
 */
class Adam : public Opt {
  public:
    // TODO
  private:
    // TODO

}; // class Adam
} // namespace mit
#endif // OPENMIT_OPTIMIZER_ADAM_H_
