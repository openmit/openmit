/*!
 *  Copyright 2016 by Contributors
 *  \file ftrl.h
 *  \brief follow the regularized leader
 *  \author ZhouYong
 */
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
    float alpha;
    /*! \brief beta per-coordinate paramater to compute learning rate */
    float beta;
    /*! \brief l1 1-norm penalty parameter for lasso */
    float l1;
    /*! \brief l2 2-norm penalty parameter for ridge */
    float l2;
    /*! \brief dim feature dimension */
    mit_uint dim;
    /*! \brief nsample_rate the rate of negative instance. [0, 1] */
    float nsample_rate;
    
    /*! \brief declare field */
    DMLC_DECLARE_PARAMETER(FtrlParam) {
      DMLC_DECLARE_FIELD(alpha).set_default(0.1);
      DMLC_DECLARE_FIELD(beta).set_default(1.0);
      DMLC_DECLARE_FIELD(l1).set_default(2);
      DMLC_DECLARE_FIELD(l2).set_default(10);
      DMLC_DECLARE_FIELD(dim).set_default(1e9);
      DMLC_DECLARE_FIELD(nsample_rate).set_default(1.0);
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
    void Update(const dmlc::Row<mit_uint> & row, 
                mit_float pred, 
                mit::SArray<mit_float> & weight_) override;

    /*! 
     * \brief unit updater for parameter server
     * \param key model feature id
     * \param idx model unit index
     * \param size model unit max size
     * \param g gradient of unit index that computed by worker node
     * \param w model parameter of unit index
     */
    void Update(const mit_uint key, 
                const uint32_t idx, 
                const uint32_t size, 
                const mit_float g, 
                mit_float & w) override;
    
  protected:
    /*! \brief parameter for ftrl optimizer */
    FtrlParam param_;
    /*! \brief z middle weight for iteration for mpi */
    mit::SArray<mit_float> zv_;
    /*! \brief n squared-sum weight for each features for mpi */
    mit::SArray<mit_float> nv_;
    /*! \brief z[i] middle weight for iteration for ps */
    PMAPT zm_;
    /*! \brief n[i] squared-sum for each features for ps */
    PMAPT nm_;

}; // class Ftrl

DMLC_REGISTER_PARAMETER(FtrlParam);

Ftrl::Ftrl(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  zv_.resize(param_.dim + 1, 0.0f);
  nv_.resize(param_.dim + 1, 0.0f);
  // TODO
}

Ftrl::~Ftrl() {
  zv_.clear(); zv_.resize(0, 0.0f);
  nv_.clear(); nv_.resize(0, 0.0f);
}

void Ftrl::Update(
    const dmlc::Row<mit_uint> & row, 
    mit_float pred, 
    mit::SArray<mit_float> & weight_) {
  // update zv_(z) and nv_(n)
  for (auto j = 0u; j < row.length; ++j) {
    mit_uint idx = row.get_index(j);
    mit_float value = row.get_value(j);
    // gradient 
    auto g = (pred - row.get_label()) * value;
    auto sigma = (sqrt(nv_[idx] + g * g) - sqrt(nv_[idx])) / param_.alpha;
    zv_[idx] += g - sigma * weight_[idx];
    nv_[idx] += g * g;
    
    int sign = zv_[idx] < 0 ? -1.0 : 1.0;

    if (sign * zv_[idx] <= param_.l1) {
      weight_[idx] = 0;
    } else {
      weight_[idx] = (param_.l1 * sign - zv_[idx]) / 
        ((param_.beta + sqrt(nv_[idx])) / param_.alpha + param_.l2);
    }
  }
}

void Ftrl::Update(const mit_uint key, 
                  const uint32_t idx, 
                  const uint32_t size, 
                  const mit_float g, 
                  mit_float & w) {
  if (nm_.find(key) == nm_.end()) {
    nm_.insert(std::make_pair(key, new mit::Unit(size)));
    zm_.insert(std::make_pair(key, new mit::Unit(size)));
  }
  auto nm_idx = nm_[key]->Get(idx);
  auto zm_idx = zm_[key]->Get(idx);
  auto sigma = (sqrt(nm_idx + g*g) - sqrt(nm_idx)) / param_.alpha;
  zm_[key]->Set(idx, zm_idx + g - sigma * w);
  nm_[key]->Set(idx, nm_idx + g*g);
  auto sign = zm_[key]->Get(idx) < 0 ? -1.0 : 1.0;
  if (sign * zm_[key]->Get(idx) <= param_.l1) {
    w = 0.0f;
  } else {
    w = (param_.l1 * sign - zm_[key]->Get(idx)) /
      ((param_.beta + sqrt(nm_[key]->Get(idx))) / param_.alpha + param_.l2);
  }
} // method Ftrl::Update

} // namespace mit
#endif // OPENMIT_OPTIMIZER_FTRL_H_
