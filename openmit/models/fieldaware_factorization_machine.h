/*!
 *  Copyright 2016 by Contributors
 *  \file fieldaware_factorization_machine.h
 *  \brief fieldaware factorization machine model
 *  \author ZhouYong, diffm
 */
#ifndef OPENMIT_MODELS_FIELDAWARE_FACTORIZATION_MACHINE_H_
#define OPENMIT_MODELS_FIELDAWARE_FACTORIZATION_MACHINE_H_

#include "openmit/models/model.h"

namespace mit {
/*!
 * \brief the field-aware factorization machine model
 */
class FFM : public Model {
  public:
    /*! \brief default constructor */
    FFM(const mit::KWArgs & kwargs) {
      this->param_.InitAllowUnknown(kwargs);
      CHECK(this->param_.field_num > 0) 
        << "param_.filed_num <= 0 for ffm model is error.";
      CHECK(this->param_.k > 0) 
        << "param_.k <= 0 for ffm model is error.";
    }

    /*! \brief destructor */
    ~FFM() {}

    /*! \brief get ffm-model pointer */
    static FFM * Get(const mit::KWArgs & kwargs) { 
      return new FFM(kwargs); 
    }

    /*! \brief prediction based on one instance */
    mit_float Predict(const dmlc::Row<mit_uint> & row, 
                      mit::PMAPT & weight,
                      bool is_norm) override;
    
    /*! \brief prediction based one instance for mpi */
    mit_float Predict(const dmlc::Row<mit_uint> & row,
                      const mit::SArray<mit_float> & weight,
                      bool is_norm) override;
    
    /*! \brief calcuate gradient based on one instance */
    void Gradient(const dmlc::Row<mit_uint> & row, 
                  const mit_float & pred,
                  mit::PMAPT & weight,
                  mit::PMAPT * grad) override;
    
    /*! \brief calculate model gradient based one instance for mpi */
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  const mit::SArray<mit_float> & weight,
                  mit::SArray<mit_float> * grad) override;

  private:
    /*! \brief ffm function expression. predict raw expression */
    mit_float RawExpr(const dmlc::Row<mit_uint> & row, 
                      mit::PMAPT & weight);

    /*! \brief ffm 1-order linear item */
    mit_float Linear(const dmlc::Row<mit_uint> & row, 
                     mit::PMAPT & weight);

    /*! \brief ffm 2-order cross item */
    mit_float Cross(const dmlc::Row<mit_uint> & row, 
                    mit::PMAPT & weight);
}; // class FFM 
} // namespace mit
#endif // OPENMIT_MODELS_FIELDAWARE_FACTORIZATION_MACHINE_H_
