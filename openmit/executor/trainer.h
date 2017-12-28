/*!
 *  Copyright (c) 2016 by Contributors
 *  \file trainer.h
 *  \brief machine learning task trainer.
 *  \author ZhouYong
 */
#ifndef OPENMIT_EXECUTOR_TRAINER_H_
#define OPENMIT_EXECUTOR_TRAINER_H_

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
#include "openmit/model/psmodel.h"

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
    /*! \brief trainer logic for ps interface with mf model*/
    void Run(std::unordered_map<ps::Key, mit::mit_float>& rating_map,
             std::vector<ps::Key> & user_keys,
             std::vector<mit_float> & user_weights,
             std::vector<int> & user_lens,
             std::vector<ps::Key> & item_keys,
             std::vector<mit_float> & item_weights,
             std::vector<int> & item_lens,
             std::vector<mit_float> * user_grads,
             std::vector<mit_float> * item_grads);

    /*! \brief metric logic for ps interface */
    void Metric(const dmlc::RowBlock<mit_uint> & batch, 
                std::vector<ps::Key> & keys, 
                std::vector<mit_float> & weights, 
                std::vector<int> & lens, 
                std::vector<float> & metrics_value);
 
    /*! \brief metric logic for ps interface with mf model*/
    void Metric(std::unordered_map<ps::Key, mit::mit_float>& rating_map,
                std::vector<ps::Key> & user_keys,
                std::vector<mit_float> & user_weights,
                std::vector<int> & user_lens,
                std::vector<ps::Key> & item_keys,
                std::vector<mit_float> & item_weights,
                std::vector<int> & item_lens,
                std::vector<float> & metrics_value);

    inline std::vector<mit::Metric *> MetricInfo() const {
      return metrics_;
    }

    /*! \brief loss */
    void Loss(const dmlc::RowBlock<mit_uint> & batch, 
              const std::vector<mit_float> * predict, 
              std::vector<mit_float> * loss);

  private:
    /*! \brief parameter */
    mit::CliParam cli_param_;
    /*! \brief model */
    mit::PSModel * model_;
    /*! \brief metric */
    std::vector<mit::Metric *> metrics_;
    /*! \brief loss function object */
    mit::Loss * loss_;
}; // class Trainer

} // namespace mit
#endif // OPENMIT_EXECUTOR_TRAINER_H_
