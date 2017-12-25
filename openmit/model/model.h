/*!
<<<<<<< HEAD
 * \brief machin learning model interface
 */
#ifndef OPENMIT_MODELS_MODEL_H_
#define OPENMIT_MODELS_MODEL_H_
=======
 *  Copyright (c) 2016 by Contributors
 *  \file model.h
 *  \brief machine learning model
 *  \author ZhouYong
 */
#ifndef OPENMIT_MODEL_MODEL_H_
#define OPENMIT_MODEL_MODEL_H_
>>>>>>> ps

#include <omp.h>
#include <string>
#include <vector>
#include "dmlc/logging.h"
#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/data.h"
#include "openmit/common/parameter.h"
#include "openmit/entry/entry_meta.h"
#include "openmit/optimizer/optimizer.h"
#include "openmit/tools/math/basic_formula.h"
#include "openmit/tools/math/prob_distr.h"

namespace mit {
<<<<<<< HEAD

typedef std::unordered_map<mit_uint, std::pair<size_t, int> > key2offset_type;

/*!
 * \brief model template for distributed machine learning framework
 */
class Model {
  public:
    /*! \brief create a model */
    static Model * Create(const mit::KWArgs & kwargs);

    /*! \brief constructor */
    Model(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    virtual ~Model() {}

  public: // predict 
    /*! \brief prediction based on data block for ps */
    void Predict(const dmlc::RowBlock<mit_uint> & batch, 
                 const std::vector<mit_float> & weights,
                 key2offset_type & key2offset,
                 std::vector<mit_float> & preds,
                 bool norm = true);

    /*! \brief prediction based on batch data for mpi */
    void Predict(const dmlc::RowBlock<mit_uint> & batch,
                 mit::SArray<mit_float> & weight, 
                 std::vector<mit_float> * preds,
                 bool norm = true);

    /*! \brief prediction based one instance for ps */
    virtual mit_float Predict(const dmlc::Row<mit_uint> & row, 
                              const std::vector<mit_float> & weights, 
                              key2offset_type & key2offset,
                              bool norm) = 0;

    /*! \brief prediction based one instance for mpi */
    virtual mit_float Predict(const dmlc::Row<mit_uint> & row, 
                              const mit::SArray<mit_float> & weight,
                              bool norm) = 0;

    virtual mit_float Predict(const std::vector<mit_float> & user_weights,
                              const size_t user_offset,
                              const std::vector<mit_float> & item_weights,
                              size_t item_offset,
                              size_t factor_len);

  public:
    /*! \brief calcuate gradient based on one instance for ps */
    void Gradient(const dmlc::RowBlock<mit_uint>& batch,
                  const std::vector<mit_float>& weights,
                  key2offset_type& key2offset,
                  std::vector<mit_float>& loss_grads,
                  std::vector<mit_float>* grads);

    /*! \brief gradient based on batch data for mpi */
    void Gradient(const dmlc::RowBlock<mit_uint> & batch,
                  std::vector<mit_float> & preds,
                  mit::SArray<mit_float> * grads);

    /*! \brief calcuate gradient based on one instance for ps */
    virtual void Gradient(const dmlc::Row<mit_uint> & row, 
                          const std::vector<mit_float> & weights,
                          key2offset_type & key2offset,
                          std::vector<mit_float> * grads,
                          const mit_float & lossgrad_value) = 0;

    /*! \brief calculate gradient based one instance for mpi */
    virtual void Gradient(const dmlc::Row<mit_uint> & row,
                          const mit_float & pred,
                          mit::SArray<mit_float> * grad) = 0;
    virtual void Gradient(const mit_float lossgrad_value,
                          const std::vector<mit_float> & user_weights,
                          const size_t user_offset,
                          const std::vector<mit_float> & item_weights,
                          const size_t item_offset,
                          const mit_uint factor_len,          
                          std::vector<mit_float> * user_grads,
                          std::vector<mit_float> * item_grads);

  public:  // method for server
    /*! \brief pull request process */
    virtual void Pull(ps::KVPairs<mit_float> & response, 
                      mit::entry_map_type * weight) = 0;

    /*! \brief model updater. Note: virtual not pure-virtual */
    virtual void Update(const ps::SArray<mit_uint> & keys, 
                        const ps::SArray<mit_float> & vals, 
                        const ps::SArray<int> & lens, 
                        mit::entry_map_type * weight);
  
=======
/*!
 * \brief machine learning model that be suitable for mpi or local
 */
class Model {
  public:
    /*! \brief constructor */
    Model(const mit::KWArgs& kwargs);

    /*! \brief destructor */
    virtual ~Model();

    /*! \brief create a model */
    static Model* Create(const mit::KWArgs& kwargs);

    /*! \brief gradient based on batch data */
    void Gradient(const dmlc::RowBlock<mit_uint>& batch,
                  std::vector<mit_float>& preds,
                  mit::SArray<mit_float>* grads);

    /*! \brief prediction based on batch data */
    void Predict(const dmlc::RowBlock<mit_uint>& batch,
                 mit::SArray<mit_float>& weight, 
                 std::vector<mit_float>* preds,
                 bool norm = true);

    /*! \brief calculate gradient based one instance */
    virtual void Gradient(const dmlc::Row<mit_uint>& row,
                          const mit_float& pred,
                          mit::SArray<mit_float>* grad) = 0;

    /*! \brief prediction based one instance */
    virtual mit_float Predict(const dmlc::Row<mit_uint>& row, 
                              const mit::SArray<mit_float>& weight,
                              bool norm) = 0;

>>>>>>> ps
  public:
    /*! \brief get model type */
    inline std::string ModelType() { return model_param_.model; }
    /*! \brief model parameter */
    inline mit::ModelParam Param() const { return model_param_; }
    /*! \brief entry meta info */
    inline mit::EntryMeta * EntryMeta() { return entry_meta_.get(); }

  protected:
    /*! \brief client parameter */
    mit::CliParam cli_param_;
    /*! \brief model parameter */
    mit::ModelParam model_param_;
    /*! \brief entry meta information */
    std::unique_ptr<mit::EntryMeta> entry_meta_;
    /*! \brief random initialize method */
    std::unique_ptr<mit::math::ProbDistr> random_;
    /*! \brief model optimizer (default) */
    std::unique_ptr<mit::Optimizer> optimizer_;
}; // class Model

} // namespace mit
<<<<<<< HEAD
#endif // OPENMIT_MODELS_MODEL_H_
=======
#endif // OPENMIT_MODEL_MODEL_H_
>>>>>>> ps
