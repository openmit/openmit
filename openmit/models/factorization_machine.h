#ifndef OPENMIT_MODELS_FACTORIZATION_MACHINE_H_
#define OPENMIT_MODELS_FACTORIZATION_MACHINE_H_

#include "openmit/models/model.h"

namespace mit {

/*!
 * \brief the factorization machine model for worker phase
 *  
 * Predict:
 *  sum = 0;
 *  1. bias:    bias += w0
 *  2. linear:  linear += wi * xi 
 *  3. cross:   cross += \frac{1}{2} \sum_{f=1}^{k}  
 *              {\left \lgroup \left(\sum_{i=1}^{n} v_{i,f} x_i \right)^2 
 *              - \sum_{i=1}^{n} v_{i,f}^2 x_i^2\right \rgroup}
 *  sum = bias + linear + cross;
 *  pred = mit::math::sigmoid(sum);
 *
 * Gradient:
 *  1. 1        if w0,
 *  2. xi       if wi (i = 1, ..., n),
 *  3. xi * (\sum_{j=1}^{n} v_{jf}*xj - v_{if} * xi)  if w_{if} (f=1,...,k). 
 */
class FM : public Model {
  public:
    /*! \brief default constructor */
    FM(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    ~FM() {}

    /*! \brief get fm model pointer */
    static FM * Get(const mit::KWArgs & kwargs) { 
      return new FM(kwargs); 
    }

    /*! 
     * \brief prediction based on one instance for ps
     *        computation complexity: O(nk)
     */
    mit_float Predict(
        const dmlc::Row<mit_uint> & row, 
        std::unordered_map<mit_uint, mit::Unit * > & map_weight,
        bool is_norm) override;
    
    /*! \brief prediction based one instance for mpi */
    mit_float Predict(
        const dmlc::Row<mit_uint> & row,
        const mit::SArray<mit_float> & weight,
        bool is_norm) override;
    
    /*! \brief calcuate gradient based on one instance for ps */
    void Gradient(
        const dmlc::Row<mit_uint> & row, 
        const mit_float & pred,
        std::unordered_map<mit_uint, mit::Unit * > & map_weight,
        std::unordered_map<mit_uint, mit::Unit * > * map_grad) override;

    /*! \brief calculate model gradient based one instance for mpi */
    void Gradient(
        const dmlc::Row<mit_uint> & row,
        const mit_float & pred,
        const mit::SArray<mit_float> & weight,
        mit::SArray<mit_float> * grad) override;

  private:
    /*! \brief fm function expression. predict raw score*/
    mit_float PredictRaw(const dmlc::Row<mit_uint> & row, 
                         std::unordered_map<mit_uint, mit::Unit * > & weight);

    /*! \brief ffm 1-order linear item */
    mit_float Linear(const dmlc::Row<mit_uint> & row, 
                     std::unordered_map<mit_uint, mit::Unit * > & weight);

    /*! \brief ffm 2-order cross item */
    mit_float Cross(const dmlc::Row<mit_uint> & row, 
                    std::unordered_map<mit_uint, mit::Unit * > & weight);

}; // class FM 

} // namespace mit

#endif // OPENMIT_MODELS_FACTORIZATION_MACHINE_H_
