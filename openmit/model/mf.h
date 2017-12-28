/*!
 *  Copyright 2016 by Contributors
 *  \file mf.h
 *  \brief matrix factorizationmodel
 *  \author ZhouYong,iamhere1
 */
#ifndef OPENMIT_MODELS_MF_H_
#define OPENMIT_MODELS_MF_H_

#include "openmit/model/model.h"
#include "openmit/model/psmodel.h"

namespace mit {
/*!
 * \brief the matrix factorization model 
 *        that be suitable for ps framework
 */
class PSMF : public PSModel {
  public:
    /*! \brief default constructor */
    PSMF(const mit::KWArgs & kwargs) :  PSModel(kwargs) {}

    /*! \brief destructor */
    virtual ~PSMF();

    /*! \brief get fm model pointer */
    static PSMF * Get(const mit::KWArgs & kwargs);

    /*! \brief pull request */
    void Pull(ps::KVPairs<mit_float> & response,
              mit::entry_map_type * weight) override;

    /*! \brief calcuate gradient based on one instance for ps */
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const std::vector<mit_float> & weights,
                  key2offset_type & key2offset,
                  std::vector<mit_float> * grads,
                  const mit_float & lossgrad_value){
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
    /*! \brief prediction based one instance for ps */
    mit_float Predict(const std::vector<mit_float> & user_weights,
                      const size_t user_offset,
                      const std::vector<mit_float> & item_weights,
                      size_t item_offset,
                      size_t factor_len);

    void SolveByAls(std::unordered_map<ps::Key, mit::mit_float>& rating_map,
                    std::vector<ps::Key>& user_keys,
                    std::vector<mit_float> & user_weights,
                    std::vector<int> & user_lens,
                    std::vector<ps::Key> & item_keys,
                    std::vector<mit_float> & item_weights,
                    std::vector<int> & item_lens,
                    std::vector<mit_float> * user_grads,
                    std::vector<mit_float> * item_grads);


  private:
    /*! \brief lr model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
}; // class MF 

} // namespace mit

#endif // OPENMIT_MODELS_MF_H_
