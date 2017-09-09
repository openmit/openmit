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

void Updater::Run(const ps::KVPairs<mit_float> * req_data,
                  PMAPT * weight) {
  // step1: req_data(keys, vals, lens) -> grad
  PMAPT grad;
  auto size = req_data->keys.size();
  auto unit_size = param_.field_num * param_.k + 1;
  auto offset = 0u;
  for (auto i = 0u; i < size; ++i) {
    CHECK_EQ(unit_size, req_data->lens[i]);
    mit::Unit * unit = new mit::Unit(unit_size);
    ps::SArray<mit_float> unitgrad = 
      req_data->vals.segment(offset, offset + req_data->lens[i]);
    CHECK_EQ(unit_size, unitgrad.size()) << "unit_size != unitgrad.size()";
    unit->CopyFrom(unitgrad.data(), unitgrad.size());
    grad.insert(std::make_pair(req_data->keys[i], unit));
    offset += req_data->lens[i];

    // initialize feature weight when not found
    if (weight->find(req_data->keys[i]) == weight->end()) {
      mit::Unit * unit = new mit::Unit(unit_size);
      weight->insert(std::make_pair(req_data->keys[i], unit));
    }
  }

  // step3: update model based on grad by callback optimizer
  Update(grad, weight);
} // method Run

void Updater::Update(PMAPT & grad, PMAPT * weight) {
  opt_->Run(grad, weight);
} // method Update

} // namespace mit
