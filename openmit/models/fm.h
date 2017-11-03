/*!
 *  Copyright 2016 by Contributors
 *  \file fm.h
 *  \brief factorization machine model
 *  \author ZhouYong, diffm
 */
#ifndef OPENMIT_MODELS_FM_H_
#define OPENMIT_MODELS_FM_H_

#include "openmit/models/model.h"

namespace mit {
/*!
 * \brief the factorization machine model in worker (data) node
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
    FM(const mit::KWArgs & kwargs) : Model(kwargs) {}

    /*! \brief destructor */
    ~FM() {}

    /*! \brief get fm model pointer */
    static FM * Get(const mit::KWArgs & kwargs) { 
      return new FM(kwargs); 
    }

  public:  // method for server
    /*! \brief initialize model optimizer */
    void InitOptimizer(const mit::KWArgs & kwargs) override;

    /*! \brief pull request */
    void Pull(ps::KVPairs<mit_float> & response, 
              mit::entry_map_type * weight) override;
 
    /*! \brief update */
    void Update(const ps::SArray<mit_uint> & keys, 
                const ps::SArray<mit_float> & vals, 
                const ps::SArray<int> & lens, 
                mit::entry_map_type * weight) override;
  
  public:
    /*! \brief calcuate gradient based on one instance for ps */
    void Gradient(const dmlc::Row<mit_uint> & row, 
                  const std::vector<mit_float> & weights,
                  mit::key2offset_type & key2offset,
                  std::vector<mit_float> * grads,
                  const mit_float & lossgrad_value) override; 

    /*! \brief calculate model gradient based one instance for mpi */
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::SArray<mit_float> * grad) override;
  
  public:
    /*! \brief prediction based one instance for ps */
    mit_float Predict(const dmlc::Row<mit_uint> & row, 
                      const std::vector<mit_float> & weights, 
                      mit::key2offset_type & key2offset, 
                      bool is_norm) override;

    /*! \brief prediction based one instance for mpi */
    mit_float Predict(const dmlc::Row<mit_uint> & row,
                      const mit::SArray<mit_float> & weight,
                      bool is_norm) override;
  
  private:
    /*! \brief fm 1-order linear item */
    mit_float Linear(const dmlc::Row<mit_uint> & row, 
                     const std::vector<mit_float> & weights, 
                     mit::key2offset_type & key2offset);

    /*! \brief fm 2-order cross item */
    mit_float Cross(const dmlc::Row<mit_uint> & row, 
                    const std::vector<mit_float> & weights, 
                    mit::key2offset_type & key2offset);

  private:
    /*! \brief lr model optimizer for w */
    std::unique_ptr<mit::Optimizer> optimizer_;   
    /*! \brief lr model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
}; // class FM 

} // namespace mit

#endif // OPENMIT_MODELS_FM_H_
