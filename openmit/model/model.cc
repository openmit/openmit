#include "openmit/model/fm.h"
#include "openmit/model/ffm.h"
#include "openmit/model/lr.h"
#include "openmit/model/mf.h"
#include "openmit/model/model.h"

namespace mit {

Model::Model(const mit::KWArgs& kwargs) {
  cli_param_.InitAllowUnknown(kwargs);
  model_param_.InitAllowUnknown(kwargs);
  entry_meta_.reset(new mit::EntryMeta(model_param_));
  random_.reset(mit::math::Random::Create(model_param_));
  optimizer_.reset(mit::Optimizer::Create(kwargs));
  LOG(INFO) << "cli_param_.num_thread: " << cli_param_.num_thread;
}

Model::~Model() {}

Model* Model::Create(const mit::KWArgs& kwargs) {
  std::string model = "lr";
  for (auto & kv : kwargs) {
    if (kv.first != "model") continue;
    model = kv.second;
  }
  if (model == "lr") {
    return mit::LR::Get(kwargs);
  } else if (model == "fm") {
    return mit::FM::Get(kwargs);
  } else if (model == "ffm") {
    return mit::FFM::Get(kwargs);
  } else if (model == "mf") {
    return mit::MF::Get(kwargs);
  } else {
    LOG(FATAL) << "unknown model. " << model;
    return nullptr;
  }
}

void Model::Predict(const dmlc::RowBlock<mit_uint>& batch, 
                      const std::vector<mit_float>& weights, 
                      key2offset_type& key2offset, 
                      std::vector<mit_float>& preds) {
  CHECK_EQ(batch.size, preds.size());
  #pragma omp parallel for num_threads(cli_param_.num_thread)
  for (auto i = 0u; i < batch.size; ++i) {
    preds[i] = Predict(batch[i], weights, key2offset);
  }
}

void Model::Gradient(const dmlc::RowBlock<mit_uint>& batch, 
                       const std::vector<mit_float>& weights, 
                       key2offset_type& key2offset, 
                       std::vector<mit_float>& loss_grads, 
                       std::vector<mit_float>* grads) {
  CHECK_EQ(weights.size(), grads->size());
  auto nthread = cli_param_.num_thread; CHECK(nthread > 0);
  #pragma omp parallel for num_threads(nthread)
  for (auto i = 0; i < batch.size; ++i) {
    Gradient(batch[i], weights, key2offset, grads, loss_grads[i]);
  }
  #pragma omp parallel for num_threads(nthread)
  for (auto i = 0u; i < grads->size(); ++i) {
    (*grads)[i] /= batch.size;
  }
  /*
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
  #pragma omp parallel for num_threads(nthread)
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
  */
} // Model::Gradient

void Model::Update(const ps::SArray<mit_uint>& keys, 
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

float Model::InnerProductWithSSE(const float* p1, const float* p2) {
  float sum = 0.0f;
  __m128 inprod = _mm_setzero_ps();
  for (auto offset = 0u; offset < blocksize; offset += 4) {
    __m128 v1 = _mm_loadu_ps(p1 + offset);
    __m128 v2 = _mm_loadu_ps(p2 + offset);
    inprod = _mm_add_ps(inprod, _mm_mul_ps(v1, v2));
  }
  inprod = _mm_hadd_ps(inprod, inprod);
  inprod = _mm_hadd_ps(inprod, inprod);
  float v;
  _mm_store_ss(&v, inprod);
  sum += v;

  for (auto i = 0u; i < remainder; ++i) {
    sum += p1[blocksize + i] * p2[blocksize + i];
  }
  return sum;
} // InnerProductWithSSE

void Model::GradientEmbeddingWithSSE(const float* pweight, 
                                     float* pgrad, 
                                     const float& value) {
  __m128 mMiddle = _mm_set1_ps(value);
  __m128 mWeight;
  __m128 mRes;
  for (auto i = 0u; i < blocksize; i += 4) {
    mWeight = _mm_loadu_ps(pweight + i);
    mRes = _mm_mul_ps(mWeight, mMiddle);
    const float* q = (const float*)&mRes;
    for (int j = 0; j < 4; ++j) pgrad[i + j] += q[j];  
  }
  
  if (remainder > 0) {
    for (auto j = 0u; j < remainder; ++j) {
      pgrad[blocksize + j] = pweight[blocksize + j] * value;
    }
  }
} // GradientEmbeddingWithSSE

} // namespace mit
