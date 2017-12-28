/*!
 *  Copyright (c) 2016 by Contributors
 *  \file fm.h
 *  \brief factorization machine model
 *  \author ZhouYong, diffm
 */
#ifndef OPENMIT_MODEL_FM_H_
#define OPENMIT_MODEL_FM_H_

#include "openmit/model/model.h"
#include "openmit/model/psmodel.h"

namespace mit {
/*!
 * \brief the factorization machine model 
 *        that be suitable for ps framework
 *  
 * Predict:
 *  sum = 0;
 *  1. intercept:    intercept += w0
 *  2. linear:  linear += wi * xi 
 *  3. cross:   cross += \frac{1}{2} \sum_{f=1}^{k}  
 *              {\left \lgroup \left(\sum_{i=1}^{n} v_{i,f} x_i \right)^2 
 *              - \sum_{i=1}^{n} v_{i,f}^2 x_i^2\right \rgroup}
 *  sum = intercept + linear + cross;
 *  pred = mit::math::sigmoid(sum);
 *
 * Gradient:
 *  1. 1                                              if w0,
 *  2. xi                                             if wi (i = 1, ..., n),
 *  3. xi * (\sum_{j=1}^{n} v_{jf}*xj - v_{if} * xi)  if w_{if} (f=1,...,k). 
 */
class FM : public Model {
  public:
    /*! \brief default constructor */
    FM(const mit::KWArgs& kwargs) : Model(kwargs) {
      optimizer_v_.reset(
        mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
    }

    /*! \brief destructor */
    virtual ~FM();

    /*! \brief get fm model pointer */
    static FM* Get(const mit::KWArgs& kwargs);

    /*! \brief model gradient based on one instance */
    void Gradient(const dmlc::Row<mit_uint>& row,
                  const mit_float& pred,
                  mit::SArray<mit_float>* grad) override;
  
    /*! \brief prediction based on one instance */
    mit_float Predict(const dmlc::Row<mit_uint>& row,
                      const mit::SArray<mit_float>& weight,
                      bool norm) override;
  
  private:
    /*! \brief fm model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
}; // class FM 

class PSFM : public PSModel {
  public:
    /*! \brief default constructor */
    PSFM(const mit::KWArgs& kwargs) : PSModel(kwargs) {
      optimizer_v_.reset(
        mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
    }

    /*! \brief destructor */
    virtual ~PSFM();

    /*! \brief get fm model pointer */
    static PSFM* Get(const mit::KWArgs& kwargs);

    /*! \brief pull request process method for server */
    void Pull(ps::KVPairs<mit_float>& response, 
              mit::entry_map_type* weight) override;
 
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

    /*! \brief updater */
    void Update(const ps::SArray<mit_uint>& keys, 
                const ps::SArray<mit_float>& vals, 
                const ps::SArray<int>& lens, 
                mit::entry_map_type* weight) override;
  
  private:
    /*! \brief fm 1-order item (linear) */
    mit_float Linear(const dmlc::Row<mit_uint>& row, 
                     const std::vector<mit_float>& weights, 
                     mit::key2offset_type& key2offset);

    /*! \brief fm 2-order item (cross) */
    mit_float Cross(const dmlc::Row<mit_uint>& row, 
                    const std::vector<mit_float>& weights, 
                    mit::key2offset_type& key2offset);

  private:
    /*! \brief fm model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
}; // class PSFM 

} // namespace mit
#endif // OPENMIT_MODEL_FM_H_
