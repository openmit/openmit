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
#include "ps/base.h"
#include "openmit/common/arg.h"
#include "openmit/common/base.h"
#include "openmit/loss/loss.h"
#include "openmit/metric/metric.h"
#include "openmit/models/model.h"

#include "openmit/loss/loss.h"

namespace mit {
/*!
 * \brief trainer template used worker node 
 *  for distributed parameter server framework
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
    void Run(const dmlc::RowBlock<mit_uint> & batch, 
             std::vector<ps::Key> & keys, 
             std::vector<mit_float> & weights, 
             std::vector<int> & lens, 
             std::vector<mit_float> * grads);

    /*! \brief metric logic for ps interface */
    void Metric(const dmlc::RowBlock<mit_uint> & batch, 
                std::vector<ps::Key> & keys, 
                std::vector<mit_float> & weights, 
                std::vector<int> & lens, 
                std::vector<float> & metrics_value);

    inline std::vector<mit::Metric *> MetricInfo() const {
      return metrics_;
    }

    /*! \brief trainer logic for mpi interface */
    // TODO
    
    /*! \brief loss */
    void Loss(const dmlc::RowBlock<mit_uint> & batch, 
              const std::vector<mit_float> * predict, 
              std::vector<mit_float> * loss);

  private:
    /*! \brief parameter */
    mit::CliParam cli_param_;
    /*! \brief model */
    mit::Model * model_;
    /*! \brief metric */
    std::vector<mit::Metric *> metrics_;
    /*! \brief loss function object */
    mit::Loss * loss_;

}; // class Trainer

} // namespace mit

#endif // OPENMIT_ENGINE_TRAINER_H_
