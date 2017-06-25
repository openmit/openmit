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

    /*! \brief get lr-model pointer */
    inline static LR * Get(const mit::KWArgs & kwargs) {
      return new LR(kwargs);
    }

    /*! \brief prediction based on one instance */
    mit_float Predict(
        const dmlc::Row<mit_uint> & row,
        std::unordered_map<mit_uint, mit::Unit * > & weight,
        bool is_norm) override;

    /*! \brief prediction based one instance for mpi */
    mit_float Predict(
        const dmlc::Row<mit_uint> & row,
        const mit::SArray<mit_float> & weight,
        bool is_norm) override;

    /*! \brief calcuate gradient based on one instance */
    void Gradient(
        const dmlc::Row<mit_uint> & row,
        const mit_float & pred,
        std::unordered_map<mit_uint, mit::Unit * > & weight,
        std::unordered_map<mit_uint, mit::Unit * > * grad) override;

    /*! \brief calculate model gradient based one instance for mpi */
    void Gradient(
        const dmlc::Row<mit_uint> & row,
        const mit_float & pred,
        const mit::SArray<mit_float> & weight,
        mit::SArray<mit_float> * grad) override;

}; // class LR

} // namespace mit

#endif // OPENMIT_MODEL_FACTORIZATION_MACHINE_H_
