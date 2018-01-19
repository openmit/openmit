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
#include "openmit/model/psmodel.h"

namespace mit {
/*!
 * \brief the field-aware factorization machine model 
 *        that be suitable for mpi or local
 */
class FFM : public Model {
  public:
    /*! \brief constructor */
    FFM(const mit::KWArgs& kwargs) : Model(kwargs) {
      optimizer_v_.reset(
        mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
    }

    /*! \brief destructor */
    virtual ~FFM();

    /*! \brief get ffm-model pointer */
    static FFM* Get(const mit::KWArgs& kwargs);

    /*! \brief calculate model gradient based one instance */
    void Gradient(const dmlc::Row<mit_uint>& row,
                  const mit_float& pred,
                  mit::SArray<mit_float>* grad) override;

    /*! \brief prediction based one instance */
    mit_float Predict(const dmlc::Row<mit_uint>& row,
                      const mit::SArray<mit_float>& weight) override;

  private:
    /*! \brief lr model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
}; // class FFM 

/*!
 * \brief the field-aware factorization machine model 
 *        that be suitable for ps framework
 */
class PSFFM : public PSModel {
  public:
    /*! 
     * \brief default constructor. 
     *        initialize custom optimizer for embedding learning
     */
    PSFFM(const mit::KWArgs& kwargs);

    /*! \brief destructor */
    virtual ~PSFFM();

    /*! \brief get ffm-model pointer */
    static PSFFM* Get(const mit::KWArgs& kwargs);

    /*! \brief pull request method for server */
    void Pull(ps::KVPairs<mit_float>& response, 
              mit::entry_map_type* weight) override;
 
    /*! \brief calcuate gradient based on one instance */
    void Gradient(const dmlc::Row<mit_uint>& row, 
                  const std::vector<mit_float>& weights,
                  mit::key2offset_type& key2offset,
                  std::vector<mit_float>* grads,
                  const mit_float& lossgrad_value) override; 

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

    /*! \brief ffm cross predict acceleration using sse instructor */
    float InProdWithSSE(const float* p1, const float* p2);

    /*! \brief ffm cross gradient acceleration using sse instructor */
    void GradEmbeddingWithSSE(const float* pweight, 
                              float* pgrad, 
                              mit_float& xij_middle);
  private:
    /*! \brief lr model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
    /*! \brief sse instruction */
    size_t blocksize = 0;
    size_t remainder = 0;
}; // class PSFFM 

} // namespace mit
#endif // OPENMIT_MODEL_FFM_H_
