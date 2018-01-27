/*!
 *  Copyright (c) 2017 by Contributors
 *  \file ffm.h
 *  \brief matrix factorization model
 *  \author WangYongJie
 */
#ifndef OPENMIT_MODEL_MF_H_
#define OPENMIT_MODEL_MF_H_

#include "openmit/model/model.h"

namespace mit {
/*! 
 * \brief matrix factorization model 
 */
class MF : public Model {
  public:
    /*! \brief constructor */
    MF(const mit::KWArgs& kwargs);
  
    /*! \brief destructor */
    virtual ~MF();

    /*! \brief get mf model */
    static MF* Get(const mit::KWArgs& kwargs);

    /*! \brief pull request processing */
    void Pull(ps::KVPairs<mit_float>& response, 
              mit::entry_map_type* weight) override;

    /*! \brief calcuate gradient based one instance */
    void Gradient(const dmlc::Row<mit_uint>& row, 
                  const std::vector<mit_float>& weights,
                  mit::key2offset_type& key2offset,
                  std::vector<mit_float>* grads,
                  const mit_float& loss_grad) override; 

    /*! \brief prediction based one instance */
    mit_float Predict(const dmlc::Row<mit_uint>& row, 
                      const std::vector<mit_float>& weights, 
                      mit::key2offset_type& key2offset) override;
}; // class MF

} // namespace mit
#endif // OPENMIT_MODEL_MF_H_
