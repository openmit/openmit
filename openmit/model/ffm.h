/*!
 *  Copyright (c) 2017 by Contributors
 *  \file ffm.h
 *  \brief field-aware factorization machine model
 *  \author ZhouYong, diffm
 */
#ifndef OPENMIT_MODEL_FFM_H_
#define OPENMIT_MODEL_FFM_H_

#include <pmmintrin.h>
#include "openmit/model/model.h"

namespace mit {
/*!
 * \brief the field-aware factorization machine model 
 *        that be suitable for ps framework
 */
class FFM : public Model {
  public:
    /*! 
     * \brief default constructor. 
     *        initialize custom optimizer for embedding learning
     */
    FFM(const mit::KWArgs& kwargs);

    /*! \brief destructor */
    virtual ~FFM();

    /*! \brief get ffm-model pointer */
    static FFM* Get(const mit::KWArgs& kwargs);

    /*! \brief pull request method for server */
    void Pull(ps::KVPairs<mit_float>& response, 
              mit::entry_map_type* weight) override;
 
    /*! \brief calcuate gradient based one instance */
    void Gradient(const dmlc::Row<mit_uint>& row, 
                  const std::vector<mit_float>& weights,
                  mit::key2offset_type& key2offset,
                  std::vector<mit_float>* grads,
                  const mit_float& loss_grad) override; 

    /*! \brief prediction based one instance */
    mit_float Predict(const dmlc::Row<mit_uint>& row, 
                      const std::vector<mit_float>& weights, 
                      mit::key2offset_type& key2offset) override;

    /*! \brief update */
    void Update(const ps::SArray<mit_uint>& keys, 
                const ps::SArray<mit_float>& vals, 
                const ps::SArray<int>& lens, 
                mit::entry_map_type* weight) override;

  private:
    /*! \brief ffm 1-order linear item */
    mit_float Linear(const dmlc::Row<mit_uint> & row, 
                     const std::vector<mit_float> & weights, 
                     mit::key2offset_type & key2offset);

    /*! \brief ffm 2-order cross item */
    mit_float Cross(const dmlc::Row<mit_uint> & row, 
                    const std::vector<mit_float> & weights, 
                    mit::key2offset_type & key2offset);

  private:
    /*! \brief lr model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
}; // class FFM 

} // namespace mit
#endif // OPENMIT_MODEL_FFM_H_
