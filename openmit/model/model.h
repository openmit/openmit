/*!
 *  Copyright (c) 2016 by Contributors
 *  \file model.h
 *  \brief machine learning model
 *  \author ZhouYong
 */
#ifndef OPENMIT_MODEL_MODEL_H_
#define OPENMIT_MODEL_MODEL_H_

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
#endif // OPENMIT_MODEL_MODEL_H_
