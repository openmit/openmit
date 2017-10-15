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
    FFM(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    ~FFM();

    /*! \brief get ffm-model pointer */
    static FFM * Get(const mit::KWArgs & kwargs) { 
      return new FFM(kwargs); 
    }

  public:  // method for server
    /*! \brief initialize model optimizer */
    void InitOptimizer(const mit::KWArgs & kwargs) override;

    /*! \brief pull request */
    void Pull(ps::KVPairs<mit_float> & response, 
              mit::EntryMeta * entry_meta, 
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
                  std::unordered_map<mit_uint, std::pair<size_t, int> > & key2offset,
                  const mit_float & preds, 
                  std::vector<mit_float> * grads) override;

    /*! \brief calculate model gradient based one instance for mpi */
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::SArray<mit_float> * grad) override;

  public:
    /*! \brief prediction based one instance for ps */
    mit_float Predict(const dmlc::Row<mit_uint> & row, 
                      const std::vector<mit_float> & weights, 
                      std::unordered_map<mit_uint, std::pair<size_t, int> > & key2offset, 
                      bool is_norm) override;

    /*! \brief prediction based one instance for mpi */
    mit_float Predict(const dmlc::Row<mit_uint> & row,
                      const mit::SArray<mit_float> & weight,
                      bool is_norm) override;

  private:
    /*! \brief ffm function expression. predict raw expression */
    //mit_float RawExpr(const dmlc::Row<mit_uint> & row, 
    //                  mit::PMAPT & weight);

    /*! \brief ffm 1-order linear item */
    //mit_float Linear(const dmlc::Row<mit_uint> & row, 
    //                 mit::PMAPT & weight);

    /*! \brief ffm 2-order cross item */
    //mit_float Cross(const dmlc::Row<mit_uint> & row, 
    //                mit::PMAPT & weight);

  private:
    /*! \brief lr model optimizer for w */
    std::unique_ptr<mit::Optimizer> optimizer_;
    
    /*! \brief lr model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;

}; // class FFM 
} // namespace mit
#endif // OPENMIT_MODELS_FIELDAWARE_FACTORIZATION_MACHINE_H_
