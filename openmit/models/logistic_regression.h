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
    LR(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    ~LR() {}

    /*! \brief get lr model */
    inline static LR * Get(const mit::KWArgs & kwargs) {
      return new LR(kwargs);
    }

  public:
    /*! \brief pull request */
    void Pull(ps::KVPairs<mit_float> & response, 
              mit::EntryMeta * entry_meta, 
              std::unordered_map<ps::Key, mit::Entry *> * weight) override;
 
    /*! \brief initialize model optimizer */
    void InitOptimizer(const mit::KWArgs & kwargs) override;

    /*! \brief update */
    void Update(const ps::SArray<mit_uint> & keys, 
                const ps::SArray<mit_float> & vals, 
                const ps::SArray<int> & lens, 
                std::unordered_map<mit_uint, mit::Entry *> * weight) override;

  private:
    /*! \brief lr model optimizer for w */
    std::unique_ptr<mit::Optimizer> optimizer_;

  public:
    /*! \brief prediction based one instance for ps */
    mit_float Predict(const dmlc::Row<mit_uint> & row, 
                      const std::vector<mit_float> & weights, 
                      std::unordered_map<mit_uint, std::pair<size_t, int> > & key2offset, 
                      bool is_norm) override;
    
    /*! \brief calcuate gradient based on one instance for ps */
    void Gradient(const dmlc::Row<mit_uint> & row, 
                  const std::vector<mit_float> & weights,
                  std::unordered_map<mit_uint, std::pair<size_t, int> > & key2offset,
                  const mit_float & preds, 
                  std::vector<mit_float> * grads) override;

    /*! \brief prediction based one instance for mpi */
    mit_float Predict(const dmlc::Row<mit_uint> & row,
                      const mit::SArray<mit_float> & weight,
                      bool is_norm) override;

    /*! \brief calculate model gradient based one instance for mpi */
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::SArray<mit_float> * grad) override;
}; // class LR
} // namespace mit

#endif // OPENMIT_MODEL_FACTORIZATION_MACHINE_H_
