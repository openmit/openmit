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
    void Update(const dmlc::Row<mit_uint> & row, 
                mit_float pred, 
                mit::SArray<mit_float> & weight_) override;

    /*! \brief parameter updater for ps */
    void Update(PMAPT & map_grad, PMAPT * weight) override;
    
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

//void Ftrl::Update(std::unordered_map<ps::Key, mit::Unit * > & grad,
//                  std::unordered_map<ps::Key, mit::Unit * > * weight) {
void Ftrl::Update(PMAPT & grad, PMAPT * weight) {
  // update z and n
  for (const auto & kunit : grad) {
    auto feati = kunit.first;
    mit::Unit * unit = kunit.second;
    auto size = unit->Size();
    CHECK(size >= 1) << "length of unit should not less than 1.";
    
    if (nm_.find(feati) == nm_.end()) {
      nm_.insert(std::make_pair(feati, new mit::Unit(size)));
    }
    if (zm_.find(feati) == zm_.end()) {
      zm_.insert(std::make_pair(feati, new mit::Unit(size)));
    }

    // not support fm/ffm cross item 
    for (auto idx = 0u; idx < size; ++idx) {
      auto g = grad[feati]->Get(idx);
      auto nm_idx = nm_[feati]->Get(idx);
      auto zm_idx = zm_[feati]->Get(idx);
      auto w_idx = (*weight)[feati]->Get(idx);

      auto sigma = (sqrt(nm_idx + g * g) - sqrt(nm_idx)) / param_.alpha;
      zm_[feati]->Set(idx, zm_idx + g - sigma * w_idx);
      nm_[feati]->Set(idx, nm_idx + g * g);

      if (idx == 0) {  // update w : ftrl for linear item 
        auto sign = zm_[feati]->Get(idx) < 0 ? -1.0f : 1.0f;
        if (sign * zm_[feati]->Get(idx) <= param_.l1) {
          (*weight)[feati]->Set(idx, 0);
        } else {
          auto updated_w = (param_.l1 * sign - zm_[feati]->Get(idx)) / 
            ((param_.beta + sqrt(nm_[feati]->Get(idx))) / param_.alpha + param_.l2);
          (*weight)[feati]->Set(idx, updated_w);
        } 
      } else {  // update v : adagrad for cross item 
        (*weight)[feati]->Set(idx, w_idx - param_.lrate * g / sqrt(nm_[feati]->Get(idx)));
      }
    }
  }
} // method Ftrl::Update
} // namespace mit
#endif // OPENMIT_OPTIMIZER_FTRL_H_
