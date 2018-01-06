/*!
 *  Copyright (c) 2017 by Contributors
 *  \file psmodel.h
 *  \brief machine learning model for parameter server
 *  \author ZhouYong
 */
#ifndef OPENMIT_MODEL_PSMODEL_H_
#define OPENMIT_MODEL_PSMODEL_H_

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

namespace mit {
/*! \brief define variable for key to offset */
typedef std::unordered_map<
  mit_uint, std::pair<size_t, int> > key2offset_type;
/*!
 * \brief machine learning model based parameter server framework
 */
class PSModel {
  public:
    /*! \brief constructor */
    PSModel(const mit::KWArgs& kwargs);

    /*! \brief destructor */
    virtual ~PSModel();
    
    /*! \brief create a ps model */
    static PSModel* Create(const mit::KWArgs& kwargs);

    /*! 
     * \brief predictor based on batch data (row block)
     * \param batch multiple instance 
     * \param weights model parameter by pull op from server 
     * \param key2offset map between key and value offset 
     * \param preds prediction result saved
     * \param norm whether normalization 
     */
    void Predict(const dmlc::RowBlock<mit_uint>& batch, 
                 const std::vector<mit_float>& weights,
                 key2offset_type& key2offset,
                 std::vector<mit_float>& preds,
                 bool norm = true);

    /*! 
     * \brief prediction based on one instance (pure-virtual)
     * \param row one instance 
     * \param weights model parameter by pull op from server 
     * \param key2offset map between key and value offset 
     * \param norm whether normalization 
     * \return prediction result
     */
    virtual mit_float Predict(const dmlc::Row<mit_uint>& row,
                              const std::vector<mit_float>& weights, 
                              key2offset_type& key2offset,
                              bool norm) = 0;

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
}; // class PSModel

} // namespace mit
#endif // OPENMIT_MODEL_PSMODEL_H_
