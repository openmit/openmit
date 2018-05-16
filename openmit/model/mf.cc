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
  mit_uint index_threashold = ((mit_uint)1 << cli_param_.nbit);
  mit_uint userkey = row.index[0];
  mit_uint itemkey = row.index[1];
  CHECK_LT(userkey, index_threashold);
  CHECK_LT(itemkey, index_threashold);
  auto key1_offset = key2offset[userkey].first;
  auto key2_offset = key2offset[NewKey(1, itemkey, cli_param_.nbit)].first;
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
  mit_uint index_threashold = ((mit_uint)1 << cli_param_.nbit);
  mit_uint userkey = row.index[0];
  mit_uint itemkey = row.index[1];
  CHECK_LT(userkey, index_threashold);
  CHECK_LT(itemkey, index_threashold);
  auto key1_offset = key2offset[userkey].first;
  auto key2_offset = key2offset[NewKey(1, itemkey, cli_param_.nbit)].first;
  wTx = InnerProductWithSSE(weights.data() + key1_offset, weights.data() + key2_offset);
  if (cli_param_.debug) {
    LOG(INFO) << "key1_offset:" << key1_offset << " key2_offset:" << key2_offset << " res:" << wTx;
  }
  return wTx;
}

}// namespace mit
