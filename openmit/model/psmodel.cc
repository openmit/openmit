#include "openmit/model/fm.h"
#include "openmit/model/ffm.h"
#include "openmit/model/linear_reg.h"
#include "openmit/model/mf.h"
#include "openmit/model/psmodel.h"

namespace mit {

PSModel::PSModel(const mit::KWArgs& kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  model_param_.InitAllowUnknown(kwargs);
  entry_meta_.reset(new mit::EntryMeta(model_param_));
  random_.reset(mit::math::Random::Create(model_param_));
  optimizer_.reset(mit::Optimizer::Create(kwargs, cli_param_.optimizer));
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
  } else if (model == "mf") {
    return mit::PSMF::Get(kwargs);
  } else {
    LOG(FATAL) << "unknown model. " << model;
    return nullptr;
  }
}

void PSModel::Predict(const dmlc::RowBlock<mit_uint>& batch, 
                      const std::vector<mit_float>& weights, 
                      key2offset_type& key2offset, 
                      std::vector<mit_float>& preds, 
                      bool norm) {
  CHECK_EQ(batch.size, preds.size());
  #pragma omp parallel for num_threads(cli_param_.num_thread)
  for (auto i = 0u; i < batch.size; ++i) {
    preds[i] = Predict(batch[i], weights, key2offset, norm);
  }
}

mit_float PSModel::Predict(const std::vector<mit_float> & user_weights,
                           const size_t user_offset,
                           const std::vector<mit_float> & item_weights,
                           size_t item_offset,
                           size_t factor_len){
  mit_float sum = 0.0f;
  for (size_t i = 0; i < factor_len; i++){
    sum += user_weights[user_offset + i] * item_weights[item_offset + i];
  }
  return sum;
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

void PSModel::Gradient(const mit_float lossgrad_value,
                     const std::vector<mit_float> & user_weights,
                     const size_t user_offset,
                     const std::vector<mit_float> & item_weights,
                     const size_t item_offset,
                     const mit_uint factor_len,
                     std::vector<mit_float> * user_grads,
                     std::vector<mit_float> * item_grads){
  for (size_t k = 0; k < factor_len; k++) {
    (*user_grads)[user_offset + k] += lossgrad_value * item_weights[item_offset + k];
    (*item_grads)[item_offset + k] += lossgrad_value * user_weights[user_offset + k];
  }  
}

void PSModel::SolveByAls(std::unordered_map<ps::Key, mit::mit_float>& rating_map,
                         std::vector<ps::Key>& user_keys,
                         std::vector<mit_float> & user_weights,
                         std::vector<int> & user_lens,
                         std::vector<ps::Key> & item_keys,
                         std::vector<mit_float> & item_weights,
                         std::vector<int> & item_lens,
                         std::vector<mit_float> * user_res_vector,
                         std::vector<mit_float> * item_res_vector) {
}


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
        if (cli_param_.optimizer=="als") { //als optimization weight updation
          (*weight)[key]->Set(idx, g);
        }
        else { //sgd optimization weight updation
          optimizer_->Update(key, idx, g, w, (*weight)[key]);
          (*weight)[key]->Set(idx, w);
        }
      }
    }
    CHECK_EQ(offset, vals.size()) << "offset not match vals.size.";
  }
}

} // namespace mit
