#include "openmit/engine/updater.h"
#include "ps/ps.h"

namespace mit {

DMLC_REGISTER_PARAMETER(UpdaterParam);

Updater::Updater(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  Init(kwargs);
}

Updater::~Updater() {
  // TODO
}

void Updater::Init(const mit::KWArgs & kwargs) {
  opt_.reset(mit::Opt::Create(kwargs, param_.optimizer));
}

void Updater::Run(
    const ps::KVPairs<mit_float> * req_data,
    std::unordered_map<ps::Key, mit::Unit * > * weight) {
  // step1: req_data(keys, vals, lens) -> map_grad
  std::unordered_map<ps::Key, mit::Unit * > map_grad;
  auto size = req_data->keys.size();
  auto unit_size = param_.field_num * param_.k + 1;
  auto offset = 0u;
  for (auto i = 0u; i < size; ++i) {
    CHECK_EQ(unit_size, req_data->lens[i]);
    mit::Unit * unit = new mit::Unit(unit_size);
    ps::SArray<mit_float> grad = 
      req_data->vals.segment(offset, offset + req_data->lens[i]);
    CHECK_EQ(unit_size, grad.size()) << "unit_size != grad.size()";
    unit->CopyFrom(grad.data(), grad.size());
    map_grad.insert(std::make_pair(req_data->keys[i], unit));
    offset += req_data->lens[i];

    // initialize feature weight when not found
    if (weight->find(req_data->keys[i]) == weight->end()) {
      mit::Unit * unit = new mit::Unit(unit_size);
      weight->insert(std::make_pair(req_data->keys[i], unit));
    }
  }

  // step3: optimization algorithm to update parameter base map_grad
  Update(map_grad, weight);
} // method Run

void Updater::Update(
    std::unordered_map<ps::Key, mit::Unit * > & map_grad,
    std::unordered_map<ps::Key, mit::Unit * > * weight) {
  opt_->Update(map_grad, weight);
} // method Update

} // namespace mit
