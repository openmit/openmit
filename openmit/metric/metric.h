/*!
 * Copyright 2016 by Contributors
 * \file metric.h
 * \brief interface of evalution metric function supported in openmit
 * \author ZhouYong
 */
#ifndef OPENMIT_METRIC_METRIC_H_
#define OPENMIT_METRIC_METRIC_H_

#include <string>
#include <vector>
#include "dmlc/logging.h"

namespace mit {
/*!
 * \brief interface of evaluation metric used to evaluate model performance
 *        This has nothing to do with training, 
 *        but merely act as evaluation purpose.
 */
class Metric {
  public:
    /*!
     * \brief create a metric according to name
     * \param name name of the metric
     * \return the created metric.
     */
    static Metric * Create(std::string & name);
    /*! \brief virtual destructor */
    virtual ~Metric() {}
    /*! 
     * \brief evaluate a specific metric
     * \param preds prediction
     * \param labels instance label
     */
    virtual float Eval(const std::vector<float> & preds,
                       const std::vector<float> & labels) const = 0;
    /*! \brief name of metric */
    virtual const char * Name() const = 0;

}; // class Metric
} // namespace mit

#endif // OPENMIT_METRIC_METRIC_H_
