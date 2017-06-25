/*!
 *  Copyright (c) 2016 by Contributors
 *  \file trainer.h
 *  \brief machine learning task trainer.
 *  \author ZhouYong
 */
#ifndef OPENMIT_ENGINE_TRAINER_H_
#define OPENMIT_ENGINE_TRAINER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dmlc/logging.h"

#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/common/data/data.h"
#include "openmit/entity/unit.h"
#include "openmit/metric/metric.h"
#include "openmit/models/model.h"
#include "openmit/optimizer/optimizer.h"

#include "openmit/loss/loss.h"

namespace mit {

class TrainerParam : public dmlc::Parameter<TrainerParam> {
  public:
    std::string model_type;
    std::string metric;
    std::string loss_type;
    float nsample_rate;

    DMLC_DECLARE_PARAMETER(TrainerParam) {
      DMLC_DECLARE_FIELD(model_type).set_default("lr");
      DMLC_DECLARE_FIELD(metric).set_default("logloss");
      DMLC_DECLARE_FIELD(loss_type).set_default("logit");
      DMLC_DECLARE_FIELD(nsample_rate).set_default(1.0f);
    }
}; // class TrainerParam

/*!
 * \brief trainer template for distributed machine learning framework
 */
class Trainer {
  public:
    /*! \brief constructor */
    explicit Trainer(const mit::KWArgs & kwargs);

    /*! \brief destructor */
    ~Trainer() {}

    /*! \brief initialize */
    void Init(const mit::KWArgs & kwargs);

    /*! \brief trainer logic for ps interface */
    void Run(
        const dmlc::RowBlock<mit_uint> & batch,
        std::vector<ps::Key> & keys,
        std::vector<mit_float> & rets,
        std::vector<mit_float> * vals);

    /*! \brief trainer logic for mpi interface */
    // TODO

 
    /*! \brief evaluation effect based on test set */
    float Eval(mit::DMatrix * data,
               std::vector<ps::Key> & keys,
               std::vector<mit_float> & rets);

    /*! \brief loss */
    void Loss(
        const dmlc::RowBlock<mit_uint> & batch, 
        const std::vector<mit_float> * predict, 
        std::vector<mit_float> * loss);

  protected:
    /*! \brief parameter */
    mit::TrainerParam param_;
    /*! \brief model */
    mit::Model * model_;
    /*! \brief model optimizer */
    mit::Opt * opt_;
    /*! \brief metric */
    mit::Metric * metric_;
    /*! \brief loss function */
    //std::shared_ptr<mit::Loss> loss_;
    mit::Loss * loss_;

}; // class Trainer

} // namespace mit

#endif // OPENMIT_ENGINE_TRAINER_H_
