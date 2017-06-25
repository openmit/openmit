#ifndef OPENMIT_OPTIMIZER_FTRL_H_
#define OPENMIT_OPTIMIZER_FTRL_H_

#include "dmlc/parameter.h"
#include "ps/ps.h"

#include "openmit/common/base.h"
#include "openmit/optimizer/optimizer.h"
#include "openmit/tools/dstruct/sarray.h"
using namespace mit;

namespace mit {
/*!
 * \brief FTRL-proximal algorithm parameter
 */
class FtrlParam : public dmlc::Parameter<FtrlParam> {
  public:
    /*! \brief alpha per-coordinate parameter to compute learning rate */
    mit_float alpha;
    /*! \brief beta per-coordinate paramater to compute learning rate */
    mit_float beta;
    /*! \brief l1 1-norm penalty parameter for lasso */
    mit_float l1;
    /*! \brief l2 2-norm penalty parameter for ridge */
    mit_float l2;
    /*! \brief dim feature dimension */
    mit_uint dim;
    /*! \brief nsample_rate the rate of negative instance. [0, 1] */
    mit_float nsample_rate;
    /*! \brief learning rate for adagrad */
    mit_float lrate;
    // declare
    DMLC_DECLARE_PARAMETER(FtrlParam) {
      DMLC_DECLARE_FIELD(alpha).set_default(0.1);
      DMLC_DECLARE_FIELD(beta).set_default(1.0);
      DMLC_DECLARE_FIELD(l1).set_default(2);
      DMLC_DECLARE_FIELD(l2).set_default(10);
      DMLC_DECLARE_FIELD(dim).set_default(1e9);
      DMLC_DECLARE_FIELD(nsample_rate).set_default(1.0);
      DMLC_DECLARE_FIELD(lrate).set_default(0.01);
    }
}; // class FtrlParam
/*!
 * \brief ftrl-proximal optimizer
 */
class Ftrl : public Opt {
  public:
    /*! \brief default constructor */
    Ftrl(const mit::KWArgs & kwargs);
    /*! \brief destructor */
    ~Ftrl();
    /*! \brief get a ftrl optimizer */
    static Ftrl * Get(const mit::KWArgs & kwargs) {
      return new Ftrl(kwargs);
    }
    /*! \brief parameter updater for mpi */
    void Update(
        const dmlc::Row<mit_uint> & row, 
        mit_float pred, 
        mit::SArray<mit_float> & weight_) override;
    /*! \brief parameter updater for ps */
    void Update(
        std::unordered_map<ps::Key, mit::Unit * > & map_grad,
        std::unordered_map<ps::Key, mit::Unit * > * weight) override;
    
  protected:
    /*! \brief parameter for ftrl optimizer */
    FtrlParam param_;
    /*! \brief z middle weight for iteration for mpi */
    mit::SArray<mit_float> zv_;
    /*! \brief n squared-sum weight for each features for mpi */
    mit::SArray<mit_float> nv_;
    /*! \brief z[i] middle weight for iteration for ps */
    std::unordered_map<mit_uint, mit::Unit * > zm_;
    /*! \brief n[i] squared-sum for each features for ps */
    std::unordered_map<mit_uint, mit::Unit * > nm_;

}; // class Ftrl
} // namespace mit

#endif // OPENMIT_OPTIMIZER_FTRL_H_
