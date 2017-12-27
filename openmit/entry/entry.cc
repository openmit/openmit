#include "openmit/entry/entry.h"

namespace mit {

Entry* Entry::Create(const mit::ModelParam& model_param, 
                     mit::EntryMeta* entry_meta, 
                     mit::math::Random* distr, 
                     mit_uint field) {
  if (model_param.model == "lr") {
    return new mit::LREntry(model_param, entry_meta, distr);
  } else if (model_param.model == "fm") {
    return new mit::FMEntry(model_param, entry_meta, distr);
  } else if (model_param.model == "ffm") {
    return new mit::FFMEntry(model_param, entry_meta, distr, field);
  } else {
    LOG(FATAL) << "unknown model. " << model_param.model;
    return nullptr;
  }
}

} // namespace mit
