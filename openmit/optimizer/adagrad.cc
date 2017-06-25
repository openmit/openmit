#include "openmit/optimizer/adagrad.h"

namespace mit {

DMLC_REGISTER_PARAMETER(AdaGradParam);

AdaGrad::AdaGrad(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  // TODO
}

AdaGrad::~AdaGrad() {
  // TODO
}

void AdaGrad::Update(
    const dmlc::Row<mit_uint> & row, 
    mit_float pred, 
    mit::SArray<mit_float> & weight) {
  // TODO
}

void AdaGrad::Update(std::unordered_map<ps::Key, mit::Unit * > & map_grad,
                     std::unordered_map<ps::Key, mit::Unit * > * weight) {
  // TODO
}
} // namespace mit
