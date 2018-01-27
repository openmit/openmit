#include "openmit/model/mf.h"

namespace mit {

MF::MF(const mit::KWArgs& kwargs) : Model(kwargs) {
  CHECK(model_param_.embedding_size > 0);
  this->blocksize = model_param_.embedding_size / 4 * 4;
  this->remainder = model_param_.embedding_size % 4;
} 

MF::~MF() {}

MF* MF::Get(const mit::KWArgs& kwargs) {
  return new MF(kwargs);
}

void MF::Pull(ps::KVPairs<mit_float>& response, mit::entry_map_type* weight) {
  size_t keys_size = response.keys.size(); CHECK(keys_size > 0);
  response.lens.resize(keys_size, model_param_.embedding_size);

  // key (multi-thread)
  auto nthread = cli_param_.num_thread; CHECK(nthread > 0);
  int chunksize = keys_size / nthread;
  if (keys_size % nthread != 0) chunksize += 1;
  std::vector<std::vector<mit_float>* > vals_thread(nthread);
  for (auto i = 0u; i < nthread; ++i) {
    vals_thread[i] = new std::vector<mit_float>();
  }
  #pragma omp parallel for num_threads(nthread) schedule(static, chunksize)
  for (auto i = 0u; i < keys_size; ++i) {
    int threadid = omp_get_thread_num();
    ps::Key key = response.keys[i];
    mit::Entry* entry = nullptr;
    if (weight->find(key) == weight->end()) {
      entry = mit::Entry::Create(model_param_, entry_meta_.get(), random_.get());
      CHECK_NOTNULL(entry);
      #pragma omp critical
      {
        weight->insert(std::make_pair(key, entry));
      }
    } else {
      entry = (*weight)[key];
    }
    CHECK_NOTNULL(entry); CHECK_GT(entry->Size(), 0);
    vals_thread[threadid]->insert(
        vals_thread[threadid]->end(), entry->Data(), entry->Data() + entry->Size());
  }

  // merge multi-thread result 
  for (auto i = 0u; i < nthread; ++i) {
    ps::SArray<mit_float> sarray(vals_thread[i]->data(), vals_thread[i]->size());
    response.vals.append(sarray);
    // free memory 
    delete vals_thread[i]; vals_thread[i] = nullptr;
  }
}

void MF::Gradient(const dmlc::Row<mit_uint>& row, 
                  const std::vector<mit_float>& weights, 
                  mit::key2offset_type& key2offset, 
                  std::vector<mit_float>* grads, 
                  const mit_float& loss_grad) {
  CHECK_EQ(row.length, 2) << "row format error. rating key1 key2";
  auto middle = loss_grad * row.get_weight();
  auto key1 = row.index[0];
  auto key2 = row.index[1];
  auto key1_offset = key2offset[key1].first;
  auto key2_offset = key2offset[key2].first;
  // gradient for key2
  GradientEmbeddingWithSSE(weights.data() + key1_offset, grads->data() + key2_offset, middle);
  // gradient for key1
  GradientEmbeddingWithSSE(weights.data() + key2_offset, grads->data() + key1_offset, middle);
}

mit_float MF::Predict(const dmlc::Row<mit_uint>& row, 
                      const std::vector<mit_float>& weights, 
                      mit::key2offset_type& key2offset) {
  mit_float wTx = 0.0f;
  CHECK_EQ(row.length, 2) << "row format error. rating key1 key2";
  auto key1_offset = key2offset[row.index[0]].first;
  auto key2_offset = key2offset[row.index[1]].first;
  wTx = InnerProductWithSSE(weights.data() + key1_offset, weights.data() + key2_offset);
  return wTx;
}

}// namespace mit
