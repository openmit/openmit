/*!
 *  Copyright (c) 2017 by Contributors
 *  \file model.h
 *  \brief machine learning model for parameter server
 *  \author ZhouYong
 */
#ifndef OPENMIT_MODEL_MODEL_H_
#define OPENMIT_MODEL_MODEL_H_

#include <mutex>
#include <omp.h>
#include <string>
#include <vector>
#include "dmlc/logging.h"
#include "ps/kv_app.h"
#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/data.h"
#include "openmit/common/parameter.h"
#include "openmit/common/type.h"
#include "openmit/entry/entry_meta.h"
#include "openmit/optimizer/optimizer.h"
#include "openmit/tools/math/formula.h"
#include "openmit/tools/math/random.h"
#include "openmit/loss/loss.h"

//#include "third_party/include/liblbfgs/lbfgs.h"


namespace mit {
/*! \brief define variable for key to offset */
typedef std::unordered_map<mit_uint, std::pair<size_t, int> > key2offset_type;
/*!
 * \brief machine learning model based parameter server framework
 */
class Model {
  public:
    /*! \brief constructor */
    Model(const mit::KWArgs& kwargs);

    /*! \brief destructor */
    virtual ~Model();
    
    /*! \brief create a ps model */
    static Model* Create(const mit::KWArgs& kwargs);

    /*! 
     * \brief predictor based on batch data (row block)
     * \param batch multiple instance 
     * \param weights model parameter by pull op from server 
     * \param key2offset map between key and value offset 
     * \param preds prediction result saved
     */
    void Predict(const dmlc::RowBlock<mit_uint>& batch, 
                 const std::vector<mit_float>& weights,
                 key2offset_type& key2offset,
                 std::vector<mit_float>& preds);

    /*! 
     * \brief prediction based on one instance (pure-virtual)
     * \param row one instance 
     * \param weights model parameter by pull op from server 
     * \param key2offset map between key and value offset 
     * \return prediction result
     */
    virtual mit_float Predict(const dmlc::Row<mit_uint>& row,
                              const std::vector<mit_float>& weights, 
                              key2offset_type& key2offset) = 0;

    /*! 
     * \brief model gradient based on batch data (row block)
     * \param batch multiple instance 
     * \param weights model parameter by pull op from server 
     * \param key2offset map between key and value offset 
     * \param loss_grads gradients of loss function 
     * \param grads gradients of model expr
     */
    void Gradient(const dmlc::RowBlock<mit_uint>& batch,
                  const std::vector<mit_float>& weights,
                  key2offset_type& key2offset,
                  std::vector<mit_float>& loss_grads,
                  std::vector<mit_float>* grads);

    /*! 
     * \brief calcuate model gradient based on one instance 
     * \param row one instance 
     * \param weights model parameter by pull op from server 
     * \param key2offset map between key and value offset 
     * \param grads gradients of model expr 
     * \param loss_grad loss gradient value
     */
    virtual void Gradient(const dmlc::Row<mit_uint>& row, 
                          const std::vector<mit_float>& weights,
                          key2offset_type& key2offset,
                          std::vector<mit_float>* grads,
                          const mit_float& loss_grad) = 0;

    /*! 
     * \brief pull request process applied to server 
     * \param response requested result 
     * \param weight model parameter stored in server 
     */
    virtual void Pull(ps::KVPairs<mit_float>& response, 
                      mit::entry_map_type* weight) = 0;

    /*! 
     * \brief general model updater is applicable to all parameters 
     *        that can be optimized using the same model. 
     *        for some model, it needs a custom updater. such as fm/ffm
     *        Note: virtual not pure-virtual
     * \param keys feature id array
     * \param vals feature values 
     * \param lens the value length of each feature 
     * \param weight model parameter stored in server 
     */
    virtual void Update(const ps::SArray<mit_uint>& keys, 
                        const ps::SArray<mit_float>& vals, 
                        const ps::SArray<int>& lens, 
                        mit::entry_map_type* weight);
  
    /*! \brief model type */
    inline std::string ModelType() { return model_param_.model; }

    /*! \brief model parameter */
    inline mit::ModelParam Param() const { return model_param_; }
    
    /*! \brief entry meta info */
    inline mit::EntryMeta* EntryMeta() { return entry_meta_.get(); }
   static lbfgsfloatval_t _LBFGSEvaluate(void *instance,
                                         const lbfgsfloatval_t *weights,
                                         lbfgsfloatval_t *grads,
                                         const int n,
                                         const lbfgsfloatval_t step);
    lbfgsfloatval_t LBFGSEvaluate(const lbfgsfloatval_t *weights,
                                  lbfgsfloatval_t *grads,
                                  const int n,
                                  const lbfgsfloatval_t step,
                                  const dmlc::RowBlock<mit_uint>& batch,
                                  mit::key2offset_type& key2offset,
                                  mit::Loss* loss_);
    static int _LBFGSProgress(void *instance,
                              const lbfgsfloatval_t *weights,
                              const lbfgsfloatval_t *grads,
                              const lbfgsfloatval_t fx,
                              const lbfgsfloatval_t xnorm,
                              const lbfgsfloatval_t gnorm,
                              const lbfgsfloatval_t step,
                              int n,
                              int k,
                              int ls);
    int LBFGSProgress(const lbfgsfloatval_t *x,
                      const lbfgsfloatval_t *g,
                      const lbfgsfloatval_t fx,
                      const lbfgsfloatval_t xnorm,
                      const lbfgsfloatval_t gnorm,
                      const lbfgsfloatval_t step,
                      int n,
                      int k,
                      int ls);
    void RunLBFGS(const dmlc::RowBlock<mit_uint>* batch,
                  mit::key2offset_type* key2offset,
                  mit::Loss* loss,
                  std::vector<mit_float>& weights);
  protected:
    /*! \brief inner product with sse */
    float InnerProductWithSSE(const float* p1, const float* p2);

    /*! \brief gradient embedding with sse */
    void GradientEmbeddingWithSSE(const float* pweight, 
                                  float* grads, 
                                  const float& middle);

  protected:
    /*! \brief client parameter */
    mit::CliParam cli_param_;
    /*! \brief model parameter */
    mit::ModelParam model_param_;
    /*! \brief entry meta information */
    std::unique_ptr<mit::EntryMeta> entry_meta_;
    /*! \brief random initialize method */
    std::unique_ptr<mit::math::Random> random_;
    /*! \brief model optimizer (default) */
    std::unique_ptr<mit::Optimizer> optimizer_;
    /*! \brief mutex for weight insert */
    std::mutex mu_;
    /*! \brief sse instruction */
    size_t blocksize = 0;
    size_t remainder = 0;
    const dmlc::RowBlock<mit_uint>* batch_;
    mit::key2offset_type* key2offset_;
    mit::Loss* loss_;
}; // class Model

} // namespace mit
#endif // OPENMIT_MODEL_MODEL_H_
