/*!
 *  Copyright (c) 2017 by Contributors
 *  \file linear_reg.h
 *  \brief linear regression model
 *  \author ZhouYong, diffm
 */
#ifndef OPENMIT_MODEL_LINEAR_REG_H_
#define OPENMIT_MODEL_LINEAR_REG_H_

#include "openmit/model/model.h"
#include "openmit/model/psmodel.h"

namespace mit {
/*!
 * \brief the linear regression model 
 *        that be suitable for mpi or local
 */
class LR : public Model {
  public:
    /*! \brief default constructor */
    LR(const mit::KWArgs& kwargs) : Model(kwargs) {}

    /*! \brief destructor */
    virtual ~LR();

    /*! \brief get lr model */
    static LR* Get(const mit::KWArgs& kwargs);

    /*! \brief model gradient based on one instance */
    void Gradient(const dmlc::Row<mit_uint>& row,
                  const mit_float& pred,
                  mit::SArray<mit_float>* grad) override;

    /*! \brief prediction based on one instance */
    mit_float Predict(const dmlc::Row<mit_uint>& row,
                      const mit::SArray<mit_float>& weight,
                      bool norm) override;
}; // class LR

/*!
 * \brief linear regression model 
 *        that be suitable for ps framework
 */
class PSLR : public PSModel {
  public:
    /*! \brief default constructor */
    PSLR(const mit::KWArgs& kwargs) : PSModel(kwargs) {}

    /*! \brief destructor */
    virtual ~PSLR();

    /*! \brief get lr model */
    static PSLR* Get(const mit::KWArgs& kwargs);

    /*! \brief calcuate gradient based on one instance */
    void Gradient(const dmlc::Row<mit_uint>& row, 
                  const std::vector<mit_float>& weights,
                  mit::key2offset_type& key2offset,
                  std::vector<mit_float>* grads,
                  const mit_float& loss_grad) override; 
    
    /*! \brief prediction based on one instance */
    mit_float Predict(const dmlc::Row<mit_uint>& row, 
                      const std::vector<mit_float>& weights, 
                      mit::key2offset_type& key2offset, 
                      bool norm) override;   
    
    /*! \brief pull request process */
    void Pull(ps::KVPairs<mit_float>& response, 
              mit::entry_map_type* weight) override;
}; // class PSLR

} // namespace mit
#endif // OPENMIT_MODEL_LINEAR_REG_H_
