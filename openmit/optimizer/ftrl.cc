#include "rabit/rabit.h"

#include "openmit/optimizer/ftrl.h"
#include "openmit/tools/math/basic_formula.h"

namespace mit {

DMLC_REGISTER_PARAMETER(FtrlParam);

Ftrl::Ftrl(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  zv_.resize(param_.dim + 1, 0.0f);
  nv_.resize(param_.dim + 1, 0.0f);
  // TODO
}

Ftrl::~Ftrl() {
  zv_.clear();
  nv_.clear();
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

void Ftrl::Update(std::unordered_map<ps::Key, mit::Unit * > & grad,
                  std::unordered_map<ps::Key, mit::Unit * > * weight) {
  // update z and n
  for (const auto & kunit : grad) {
    auto feati = kunit.first;
    mit::Unit * unit = kunit.second;
    auto size = unit->Size();
    
    if (nm_.find(feati) == nm_.end()) {
      //auto * unit = new mit::Unit(size);
      //nm_.insert(std::make_pair(feati, unit));
      nm_.insert(std::make_pair(feati, new mit::Unit(size)));
    }
    if (zm_.find(feati) == zm_.end()) {
      //auto * unit = new mit::Unit(size);
      //zm_.insert(std::make_pair(feati, unit));
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
