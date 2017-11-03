/*!
 *  Copyright 2016 by Contributors
 *  \file logistic_regression.h
 *  \brief logistic regression model
 *  \author ZhouYong, diffm
 */
#ifndef OPENMIT_MODEL_LOGISTIC_REGRESSION_H_
#define OPENMIT_MODEL_LOGISTIC_REGRESSION_H_

#include <memory>
#include "openmit/models/model.h"

namespace mit {
/*!
 * \brief the logistic regression model for worker phase
 */
class LR : public Model {
  public:
    /*! \brief default constructor */
    LR(const mit::KWArgs & kwargs) : Model(kwargs) {}

    /*! \brief destructor */
    ~LR() {}

    /*! \brief get lr model */
    inline static LR * Get(const mit::KWArgs & kwargs) {
      return new LR(kwargs);
    }

  public:  // method for server
    /*! \brief pull request */
    void Pull(ps::KVPairs<mit_float> & response, 
              mit::entry_map_type * weight) override;
 
    /*! \brief initialize model optimizer */
    void InitOptimizer(const mit::KWArgs & kwargs) override;

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
    /*! \brief lr model optimizer for w */
    std::unique_ptr<mit::Optimizer> optimizer_;

}; // class LR
} // namespace mit

#endif // OPENMIT_MODEL_FACTORIZATION_MACHINE_H_
