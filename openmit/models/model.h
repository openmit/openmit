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
#include "openmit/entity/entry_meta.h"
#include "openmit/optimizer/optimizer.h"
#include "openmit/tools/math/basic_formula.h"
namespace mit {
typedef std::unordered_map<mit_uint, std::pair<size_t, int> > key2offset_type;
/*!
 * \brief model template for distributed machine learning framework
 */
class Model {
  public:
    /*! \brief constructor */
    static Model * Create(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    virtual ~Model() {}

  public:
    /*! \brief prediction based on data block for ps */
    void Predict(const dmlc::RowBlock<mit_uint> & batch, 
                 const std::vector<mit_float> & weights,
                 key2offset_type & key2offset,
                 std::vector<mit_float> & preds,
                 bool is_norm = true);

    /*! \brief gradient based on data block for ps */
    void Gradient(const dmlc::RowBlock<mit_uint> & batch, 
                  const std::vector<mit_float> & weights,
                  key2offset_type & key2offset,
                  const std::vector<mit_float> & preds,
                  std::vector<mit_float> * grads);

    /*! \brief prediction based on batch data for mpi */
    void Predict(const dmlc::RowBlock<mit_uint> & batch,
                 mit::SArray<mit_float> & weight, 
                 std::vector<mit_float> * preds,
                 bool is_norm = true);

    /*! \brief gradient based on batch data for mpi */
    void Gradient(const dmlc::RowBlock<mit_uint> & batch,
                  std::vector<mit_float> & preds,
                  mit::SArray<mit_float> * grads);

  public:  // method for ps server callback 
    /*! \brief pull request */
    virtual void Pull(ps::KVPairs<mit_float> & response, 
                      mit::EntryMeta * entry_meta, 
                      std::unordered_map<ps::Key, mit::Entry *> * weight) = 0;
 
    /*! \brief initialize model optimizer */
    virtual void InitOptimizer(const mit::KWArgs & kwargs) = 0;

    /*! \brief model updater */
    virtual void Update(const ps::SArray<mit_uint> & keys, 
                        const ps::SArray<mit_float> & vals, 
                        const ps::SArray<int> & lens, 
                        std::unordered_map<mit_uint, mit::Entry *> * weight) = 0;

  public:
    /*! \brief prediction based one instance for mpi */
    virtual mit_float Predict(const dmlc::Row<mit_uint> & row, 
                              const mit::SArray<mit_float> & weight,
                              bool is_norm) = 0;

    /*! \brief calculate model gradient based one instance for mpi */
    virtual void Gradient(const dmlc::Row<mit_uint> & row,
                          const mit_float & pred,
                          mit::SArray<mit_float> * grad) = 0;

    virtual mit_float Predict(const dmlc::Row<mit_uint> & row, 
                              const std::vector<mit_float> & weights, 
                              key2offset_type & key2offset,
                              bool is_norm) = 0;
    
    /*! \brief calcuate gradient based on one instance for ps */
    virtual void Gradient(const dmlc::Row<mit_uint> & row, 
                          const std::vector<mit_float> & weights,
                          key2offset_type & key2offset,
                          const mit_float & preds, 
                          std::vector<mit_float> * grads) = 0;

    /*! \brief get model type */
    std::string ModelType() { return cli_param_.model; }
    /*! \brief model parameter */
    mit::CliParam Param() const { return cli_param_; }

  protected:
    /*! \brief model type */
    mit::CliParam cli_param_;

}; // class Model
} // namespace mit
#endif // OPENMIT_MODELS_MODEL_H_
