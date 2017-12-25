/*!
 *  Copyright 2016 by Contributors
 *  \file mf.h
 *  \brief matrix factorizationmodel
 *  \author ZhouYong,iamhere1
 */
#ifndef OPENMIT_MODELS_MF_H_
#define OPENMIT_MODELS_MF_H_

#include "openmit/model/model.h"

namespace mit {
/*!
 * \brief the matrix factorization model in worker and server (data) node
 *  
 */
class MF : public Model {
  public:
    /*! \brief default constructor */
    MF(const mit::KWArgs & kwargs) : Model(kwargs) {
      optimizer_v_.reset(mit::Optimizer::Create(
            kwargs, cli_param_.optimizer_v));
    }

    /*! \brief destructor */
    //virtual ~MF() {}
    ~MF() {}

    /*! \brief get fm model pointer */
    static MF * Get(const mit::KWArgs & kwargs) {
      return new MF(kwargs);
    }

  public:  // method for server
    /*! \brief pull request */
    void Pull(ps::KVPairs<mit_float> & response,
              mit::entry_map_type * weight) override;

    /*! \brief update */
    void Update(const ps::SArray<mit_uint> & keys,
                const ps::SArray<mit_float> & vals,
                const ps::SArray<int> & lens,
                mit::entry_map_type * weight) override;
    /*! \brief calcuate gradient based on one instance for ps */
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const std::vector<mit_float> & weights,
                  key2offset_type & key2offset,
                  std::vector<mit_float> * grads,
                  const mit_float & lossgrad_value){
    }
    /*! \brief calculate gradient based one instance for mpi */
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::SArray<mit_float> * grad){
    }

    /*! \brief calcuate gradient based on one instance for ps mf model*/
    void Gradient(const mit_float lossgrad_value,
                  const std::vector<mit_float> & user_weights,
                  const size_t user_offset,
                  const std::vector<mit_float> & item_weights,
                  const size_t item_offset,
                  const mit_uint factor_len,
                  std::vector<mit_float> * user_grads,
                  std::vector<mit_float> * item_grads);
    /*! \brief prediction based one instance for ps */
   mit_float Predict(const dmlc::Row<mit_uint> & row, 
                              const std::vector<mit_float> & weights, 
                              key2offset_type & key2offset,
                              bool norm){
     return 0.0;
   }
       /*! \brief prediction based one instance for mpi */
   mit_float Predict(const dmlc::Row<mit_uint> & row, 
                     const mit::SArray<mit_float> & weight,
                     bool norm){
     return 0.0;
   }
    /*! \brief prediction based one instance for ps */
    mit_float Predict(const std::vector<mit_float> & user_weights,
                      const size_t user_offset,
                      const std::vector<mit_float> & item_weights,
                      size_t item_offset,
                      size_t factor_len);

  private:
    /*! \brief lr model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
}; // class MF 

} // namespace mit

#endif // OPENMIT_MODELS_MF_H_
