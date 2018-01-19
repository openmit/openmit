#include "openmit/model/fm.h"
#include "openmit/model/ffm.h"
#include "openmit/model/linear_reg.h"
#include "openmit/model/psmodel.h"

namespace mit {

PSModel::PSModel(const mit::KWArgs& kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  model_param_.InitAllowUnknown(kwargs);
  entry_meta_.reset(new mit::EntryMeta(model_param_));
  random_.reset(mit::math::Random::Create(model_param_));
  optimizer_.reset(mit::Optimizer::Create(kwargs));
}

PSModel::~PSModel() {}

PSModel* PSModel::Create(const mit::KWArgs& kwargs) {
  std::string model = "lr";
  for (auto & kv : kwargs) {
    if (kv.first != "model") continue;
    model = kv.second;
  }
  if (model == "lr") {
    return mit::PSLR::Get(kwargs);
  } else if (model == "fm") {
    return mit::PSFM::Get(kwargs);
  } else if (model == "ffm") {
    return mit::PSFFM::Get(kwargs);
  } else {
    LOG(FATAL) << "unknown model. " << model;
    return nullptr;
  }
}

void PSModel::Predict(const dmlc::RowBlock<mit_uint>& batch, 
                      const std::vector<mit_float>& weights, 
                      key2offset_type& key2offset, 
                      std::vector<mit_float>& preds) {
  CHECK_EQ(batch.size, preds.size());
  #pragma omp parallel for num_threads(cli_param_.num_thread)
  for (auto i = 0u; i < batch.size; ++i) {
    preds[i] = Predict(batch[i], weights, key2offset);
  }
}

void PSModel::Gradient(const dmlc::RowBlock<mit_uint>& batch, 
                       const std::vector<mit_float>& weights, 
                       key2offset_type& key2offset, 
                       std::vector<mit_float>& loss_grads, 
                       std::vector<mit_float>* grads) {
  auto nthread = cli_param_.num_thread;
  CHECK_EQ(weights.size(), grads->size());
  std::vector<std::vector<mit_float>*> grads_thread(nthread);
  for (size_t i = 0; i < grads_thread.size(); ++i) {
    grads_thread[i] = new std::vector<mit_float>(grads->size()); 
  }
  int chunksize = batch.size / nthread;
  chunksize = batch.size % nthread == 0 ? chunksize : chunksize + 1;
  #pragma omp parallel for num_threads(nthread) schedule(static, chunksize)
  for (auto i = 0u; i < batch.size; ++i) {
    int tid = omp_get_thread_num();
    Gradient(batch[i], weights, key2offset, grads_thread[tid], loss_grads[i]);
  }
  // merge thread result to grads 
  chunksize = grads->size() / nthread;
  chunksize = grads->size() % nthread == 0 ? chunksize : chunksize + 1;
  #pragma omp parallel for num_threads(nthread) schedule(static, chunksize)
  for (auto i = 0u; i < grads->size(); ++i) {
    for (uint32_t tid = 0; tid < nthread; ++tid) {
      (*grads)[i] += (*grads_thread[tid])[i];
    }
    (*grads)[i] /= batch.size;
  }

  // free memory
  for (uint32_t i = 0; i < nthread; ++i) {
    if (grads_thread[i] != nullptr) { 
      delete grads_thread[i]; grads_thread[i] = nullptr; 
    }
  }
} // PSModel::Gradient

void PSModel::Update(const ps::SArray<mit_uint>& keys, 
                     const ps::SArray<mit_float>& vals, 
                     const ps::SArray<int>& lens, 
                     mit::entry_map_type* weight) {
  CHECK_EQ(keys.size(), lens.size());
  // for model that each feature has only one parameter, such as linear
  if (model_param_.model == "lr") {
    CHECK_EQ(keys.size(), lens.size());
    #pragma omp parallel for num_threads(cli_param_.num_thread) 
    for (auto i = 0u; i < keys.size(); ++i) {
      auto key = keys[i];
      CHECK(weight->find(key) != weight->end());
      auto w = (*weight)[key]->Get(0);
      auto g = vals[i];
      optimizer_->Update(key, 0, g, w, (*weight)[key]);
      (*weight)[key]->Set(0, w);
    }
  } else { 
    // for model that each feature has one more parameter. such as mf
    auto offset = 0u;
    for (auto i = 0u; i < keys.size(); ++i) {
      auto key = keys[i];
      CHECK(weight->find(key) != weight->end());
      auto entrysize = (*weight)[key]->Size();
      CHECK_EQ(entrysize, (size_t)lens[i]);
      // for each entry 
      for (auto idx = 0u; idx < entrysize; ++idx) {
        auto w = (*weight)[key]->Get(idx);
        auto g = vals[offset++];
        optimizer_->Update(key, idx, g, w, (*weight)[key]);
        (*weight)[key]->Set(idx, w);
      }
    }
    CHECK_EQ(offset, vals.size()) << "offset not match vals.size.";
  }
}

} // namespace mit
