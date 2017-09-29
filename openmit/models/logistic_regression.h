/*!
 *  Copyright 2016 by Contributors
 *  \file logistic_regression.h
 *  \brief logistic regression model
 *  \author ZhouYong, diffm
 */
#ifndef OPENMIT_MODEL_LOGISTIC_REGRESSION_H_
#define OPENMIT_MODEL_LOGISTIC_REGRESSION_H_

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

    /*! \brief prediction based one instance for mpi */
    mit_float Predict(const dmlc::Row<mit_uint> & row,
                      const mit::SArray<mit_float> & weight,
                      bool is_norm) override;

    /*! \brief calculate model gradient based one instance for mpi */
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::SArray<mit_float> * grad) override;

  /*! \brief prediction based on one instance for ps */
    mit_float Predict(const dmlc::Row<mit_uint> & row,
                      mit::PMAPT & weight,
                      bool is_norm) override;

    /*! \brief calcuate gradient based on one instance for ps*/
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::PMAPT & weight,
                  mit::PMAPT * grad) override;

}; // class LR
} // namespace mit

#endif // OPENMIT_MODEL_FACTORIZATION_MACHINE_H_
