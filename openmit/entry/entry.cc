#include "openmit/entry/entry.h"
#include "openmit/entry/entry_lr.h"
#include "openmit/entry/entry_fm.h"
#include "openmit/entry/entry_ffm.h"
#include "openmit/entry/entry_mf.h"

namespace mit {

Entry* Entry::Create(const mit::ModelParam& model_param, 
                     mit::EntryMeta* entry_meta, 
                     mit::math::Random* random, 
                     mit_uint field) {
  if (model_param.model == "lr") {
    return new mit::LREntry(model_param, entry_meta, random);
  } else if (model_param.model == "fm") {
    return new mit::FMEntry(model_param, entry_meta, random);
  } else if (model_param.model == "ffm") {
    return new mit::FFMEntry(model_param, entry_meta, random, field);
  } else if (model_param.model == "mf") {
    return new mit::MFEntry(model_param, entry_meta, random);
  } else {
    LOG(FATAL) << "unknown model. " << model_param.model;
    return nullptr;
  }
}

} // namespace mit
