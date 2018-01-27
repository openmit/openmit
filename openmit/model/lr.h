/*!
 *  Copyright (c) 2017 by Contributors
 *  \file lr.h
 *  \brief linear regression model
 *  \author ZhouYong, diffm
 */
#ifndef OPENMIT_MODEL_LR_H_
#define OPENMIT_MODEL_LR_H_

#include "openmit/model/model.h"

namespace mit {
/*!
 * \brief linear regression model 
 */
class LR : public Model {
  public:
    /*! \brief default constructor */
    LR(const mit::KWArgs& kwargs) : Model(kwargs) {}

    /*! \brief destructor */
    virtual ~LR();

    /*! \brief get lr model */
    static LR* Get(const mit::KWArgs& kwargs);

    /*! \brief calcuate gradient based on one instance */
    void Gradient(const dmlc::Row<mit_uint>& row, 
                  const std::vector<mit_float>& weights,
                  mit::key2offset_type& key2offset,
                  std::vector<mit_float>* grads,
                  const mit_float& loss_grad) override; 
    
    /*! \brief prediction based on one instance */
    mit_float Predict(const dmlc::Row<mit_uint>& row, 
                      const std::vector<mit_float>& weights, 
                      mit::key2offset_type& key2offset) override;
    
    /*! \brief pull request process */
    void Pull(ps::KVPairs<mit_float>& response, 
              mit::entry_map_type* weight) override;
}; // class LR

} // namespace mit
#endif // OPENMIT_MODEL_LR_H_
