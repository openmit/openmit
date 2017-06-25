/*!
 *  Copyright (c) 2016 by Contributors
 *  \file admm.h
 *  \brief alternating-direction-multipler-method framework
 *        for distributed computing
 *  \author ZhouYong
 */
#ifndef OPENMIT_FRAMEWORK_ADMM_ADMM_H_
#define OPENMIT_FRAMEWORK_ADMM_ADMM_H_

#include "openmit/common/parameter/parameter.h"
#include "openmit/learner/mi_learner.h"
#include "openmit/tools/dstruct/sarray.h"
#include "dmlc/io.h"
/*!
 * \brief
 *
 */
namespace mit {
/*!
 * \brief admm parameter with distributed (decentralized) algorithm
 */
class AdmmParam : public dmlc::Parameter<AdmmParam> {
  public:
    /*! \brief lambda objective function l1-norm parameter */
    float lambda_obj;
    /*!
     * \brief argument lagrangians factor that
     *        used to step size when dual variance updating.
     */
    float rho;
    /*! \brief dim size of global_weights. it equals to feature dimimension. */
    uint64_t dim;
    /*! declare parameter field */
    DMLC_DECLARE_PARAMETER(AdmmParam) {
      DMLC_DECLARE_FIELD(lambda_obj).set_default(1);
      DMLC_DECLARE_FIELD(rho).set_default(0.1);
      DMLC_DECLARE_FIELD(dim).set_default(1e8);
    }
};  // class AdmmParam
/*!
 * \brief admm algorithm framework
 */
class Admm : public MILearner {
  public:
    /*! \brief constructor */
    Admm(const mit::KWArgs & kwargs);
    /*! \brief destructor */
    virtual ~Admm() {}
    /*! \brief get admm object */
    inline static Admm * Get(const mit::KWArgs & kwargs);
    /*! running */
    void Run() override;

  private:
    /*! training phase by admm algorithm framework */
    void RunTrain();
    /*! predicting phase. offline predict scoring */
    void RunPredict();

  private:
    /*! \brief global model update */
    void UpdateGlobal();
    /*! \brief local model update */
    void UpdateLocal();
    /*! \brief lagrangian dual variables update */
    void UpdateDual();

  private:
    /*! \brief save model */
    void SaveModel(dmlc::Stream * fo, mit::SArray<mit_float> * data);
    /*! \brief load model */
    void LoadModel(dmlc::Stream * fi);
    /*! \brief save predict result. [label, predict] */
    void SavePredict(dmlc::Stream * fo,
                     std::vector<std::tuple<float, float> > & preds);

//  private:
//    /*! \brief model */
//    mit::Model model_;
//    /*! \brief optimizer */
//    mit::Optimizer opt_;
//    /*! \brief metric. need to support logloss/auc metric */
//    mit::Metric * metric_;
//  
  private:
    /*! \brief admm related parameter */
    AdmmParam param_;
    /*! \brief task related (from client) parameter */
    mit::CliParam cli_param_;
    /*! \brief global model used allreduce process. for lr */
    mit::SArray<mit_float> theta_;
    /*! \brief local model parameter. for lr */
    mit::SArray<mit_float> weight_;
    /*! \brief dual variables parameter */
    mit::SArray<mit_float> dual_;
}; // class Admm

inline Admm * Admm::
Get(const mit::KWArgs & kwargs) {
  return new Admm(kwargs);
}
} // namespace mit

#endif // OPENMIT_FRAMEWORK_ADMM_ADMM_H_
