/*!
 * \brief machin learning model interface
 */
#ifndef OPENMIT_MODELS_MODEL_H_
#define OPENMIT_MODELS_MODEL_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "dmlc/logging.h"

#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/data/data.h"
#include "openmit/entity/unit.h"
#include "openmit/tools/math/basic_formula.h"

namespace mit {

/*!
 * \brief model related parameter
 */
class ModelParam : public dmlc::Parameter<ModelParam> {
  public:
    /*! \brief model type */
    std::string model_type;
    /*! \brief number of field */
    mit_uint field_num;
    /*! \brief length of latent factor */
    mit_uint k;
    /*!
     * \brief whether to add linear item,
     *  This parameter is valid only for fm or ffm model.
     */
    bool is_linear;

    /*! \brief declare parameters */
    DMLC_DECLARE_PARAMETER(ModelParam) {
      DMLC_DECLARE_FIELD(model_type).set_default("lr");
      DMLC_DECLARE_FIELD(field_num).set_default(0);
      DMLC_DECLARE_FIELD(k).set_default(0);
      DMLC_DECLARE_FIELD(is_linear).set_default(true);
    }
}; // class ModelParam

/*!
 * \brief model template for distributed machine learning framework
 */
class Model {
  public:
    /*! \brief constructor */
    static Model * Create(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    virtual ~Model() {}

    /*! \brief prediction based on data block for ps */
    void Predict(const dmlc::RowBlock<mit_uint> & row_block,
                 std::unordered_map<mit_uint, mit::Unit * > & weight,
                 std::vector<mit_float> * preds,
                 bool is_norm = true);

    /*! \brief gradient based on data block for ps */
    void Gradient(
        const dmlc::RowBlock<mit_uint> & row_block,
        std::vector<mit_float> & preds,
        std::unordered_map<mit_uint, mit::Unit * > & weight,
        std::unordered_map<mit_uint, mit::Unit * > * grad);

    /*! \brief prediction based on one instance for ps */
    virtual mit_float Predict(
        const dmlc::Row<mit_uint> & row,
        std::unordered_map<mit_uint, mit::Unit * > & weight,
        bool is_norm) = 0;

    /*! \brief prediction based one instance for mpi */
    virtual mit_float Predict(
        const dmlc::Row<mit_uint> & row,
        const mit::SArray<mit_float> & weight,
        bool is_norm) = 0;

    /*! \brief calcuate gradient based on one instance for ps */
    virtual void Gradient(
        const dmlc::Row<mit_uint> & row,
        const mit_float & pred,
        std::unordered_map<mit_uint, mit::Unit * > & weight,
        std::unordered_map<mit_uint, mit::Unit * > * grad) = 0;

    /*! \brief calculate model gradient based one instance for mpi */
    virtual void Gradient(
        const dmlc::Row<mit_uint> & row,
        const mit_float & pred,
        const mit::SArray<mit_float> & weight,
        mit::SArray<mit_float> * grad) = 0;

    /*! \brief get model type */
    std::string ModelType() { return param_.model_type; }
    /*! \brief model parameter */
    ModelParam Param() const { return param_; }

  protected:
    /*! \brief model type */
    ModelParam param_;

}; // class Model
} // namespace mit
#endif // OPENMIT_MODELS_MODEL_H_
