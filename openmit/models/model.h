/*!
 * \brief machin learning model interface
 */
#ifndef OPENMIT_MODELS_MODEL_H_
#define OPENMIT_MODELS_MODEL_H_

#include <string>
#include <vector>
#include "dmlc/logging.h"
#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/data/data.h"
#include "openmit/common/parameter/cli_param.h"
#include "openmit/common/parameter/model_param.h"
#include "openmit/entity/entry_meta.h"
#include "openmit/optimizer/optimizer.h"
#include "openmit/tools/math/basic_formula.h"
#include "openmit/tools/math/prob_distr.h"

namespace mit {

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
                 bool is_norm = true);

    /*! \brief prediction based on batch data for mpi */
    void Predict(const dmlc::RowBlock<mit_uint> & batch,
                 mit::SArray<mit_float> & weight, 
                 std::vector<mit_float> * preds,
                 bool is_norm = true);

    /*! \brief prediction based one instance */
    virtual mit_float Predict(const dmlc::Row<mit_uint> & row, 
                              const std::vector<mit_float> & weights, 
                              key2offset_type & key2offset,
                              bool is_norm) = 0;

    /*! \brief prediction based one instance for mpi */
    virtual mit_float Predict(const dmlc::Row<mit_uint> & row, 
                              const mit::SArray<mit_float> & weight,
                              bool is_norm) = 0;


  public: // gradient
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

  public:  // method for server
    /*! \brief pull request */
    virtual void Pull(ps::KVPairs<mit_float> & response, 
                      mit::entry_map_type * weight) = 0;
 
    /*! \brief initialize model optimizer */
    virtual void InitOptimizer(const mit::KWArgs & kwargs) = 0;

    /*! \brief model updater */
    virtual void Update(const ps::SArray<mit_uint> & keys, 
                        const ps::SArray<mit_float> & vals, 
                        const ps::SArray<int> & lens, 
                        mit::entry_map_type * weight) = 0;

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
}; // class Model

} // namespace mit
#endif // OPENMIT_MODELS_MODEL_H_
