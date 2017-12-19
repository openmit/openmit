/*!
 *  Copyright (c) 2016 by Contributors
 *  \file trainer.h
 *  \brief machine learning task predictor.
 *  \author ZhouYong
 */
#ifndef OPENMIT_ENGINE_PREDICTOR_H_
#define OPENMIT_ENGINE_PREDICTOR_H_

namespace mit {
/*!
 * \brief predictor template for distributed machine learning framework
 */
class Predictor {
  public:
    /*! \brief constructor */
    explicit Predictor(const mit::KWArgs & kwargs);
    /*! \brief destructor */
    ~Predictor() {}
    /*! \brief predictor logic for ps interface */
    void Run(
        const dmlc::RowBlock<mit_uint> & batch,
        std::vector<ps::Key> & keys,
        std::vector<mit_float> & rets,
        std::vector<mit_float> * vals);

}; // class Predictor
} // namespace mit

#endif // OPENMIT_ENGINE_PREDICTOR_H_
