#include "openmit/optimizer/gd.h"

namespace mit {

DMLC_REGISTER_PARAMETER(GDParam);

GD::GD(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  if (param_.optimizer_type == "adagrad") {
    param_.is_alpha = false;
    // TODO initialize n_
  }
}

GD::~GD() {
  // TODO
}

void GD::Update(
    const dmlc::Row<mit_uint> & row, 
    mit_float pred, 
    mit::SArray<mit_float> & weight) {
  // TODO
}
// w = w - learning_rate * (grad(loss) + l1 * grad(l1) + l2 * grad(l2))
void GD::Update(
    std::unordered_map<ps::Key, mit::Unit * > & map_grad,
    std::unordered_map<ps::Key, mit::Unit * > * weight) {
  
  for (auto & kunit : map_grad) {
    auto feati = kunit.first;
    mit::Unit * unit = kunit.second;
    auto size = unit->Size();
    CHECK(size >= 1) << "length of unit < 1";
    
    for (auto idx = 0u; idx < size; ++idx) {
      auto w = (*weight)[feati]->Get(idx);
      auto grad_loss = map_grad[feati]->Get(idx);
      //auto nabla_w = grad_loss + param_.l1 * 1 + param_.l2 * w;
      auto nabla_w = grad_loss;
      auto updated_w = w - param_.alpha * nabla_w;
      /*
      std::cout << "Optimizer::GD: feati: " << feati 
        << ", grad_loss: " << grad_loss 
        <<  ", w: " << w 
        << ", updated_w: " << updated_w << std::endl;
        */
      (*weight)[feati]->Set(idx, updated_w);
    }
  }
}

} // namespace mit
