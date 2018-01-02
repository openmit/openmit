/*!
 *  Copyright 2017 by Contributors
 *  \file als.h
 *  \brief alternating least square(als) optimizer
 *  \author iamhere1
 */
#ifndef OPENMIT_OPTIMIZER_ALS_H_
#define OPENMIT_OPTIMIZER_ALS_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {
/*! 
 * \brief normal equation used for als
 */
struct NormalEquation {
  /*! \brief constructor*/
  NormalEquation(mit_uint k) {
    length = k;
    triK = k * (k + 1) / 2;
    ata = new mit_double[triK]();
    atb = new mit_double[k]();
    da = new mit_double[k](); 
    type = 'U';
    reset();
  }
  /*! \brief desctructor*/
  ~NormalEquation() {
    if (ata) { delete[] ata; ata = nullptr; }
    if (atb) { delete[] atb; atb = nullptr; }
    if (da) { delete[] da; da = nullptr; }
  }  
  /*! \brief copy feature vector to da array*/
  void copyToDouble(const std::vector<mit_float> & weights,
                    mit_uint offset,
                    mit_int n) {
    CHECK(n == length);
    for (mit_int i = 0; i < n; i++){
      da[i] = weights[offset + i];
    }
  }

  /*! \brief adds an rating to norm equation*/
  void add(const std::vector<mit_float> & weights,
           mit_uint offset,
           mit_int n,
           mit_float b,
           mit_double c = 1.0) {
    CHECK(c >= 0.0);
    CHECK(n == length);
    copyToDouble(weights, offset, n);
    mit_int incx = 1;
    mit_int incy = 1;
    //update ata:= c*da*da' + ata
    //LOG(INFO) << " da:" << mit::DebugStr(da,2, 15) << " c:" << c;
    //LOG(INFO) << " ata:" << mit::DebugStr(ata,3, 15);
    dspr_(&type, &n, &c, da, &incx, ata); //ata := c*da*da' + ata
    //LOG(INFO) << " ata:" << mit::DebugStr(ata,3, 15);
    if (b != 0.0) {
      mit_double alpha = c * b;
      /*if (cli_param_.debug) {
        LOG(INFO) << "length:"<< length << " alpha:" << alpha << " incx:" << incx << " incy:" << incy;
        LOG(INFO) << " da:" << mit::DebugStr(da,2, 15);
        LOG(INFO) << " atb before update:" << mit::DebugStr(atb, 10, 15); 
      }
      */
      //update atb:= da*alpha + atb
      //LOG(INFO) << " atb:" << mit::DebugStr(atb,3, 15);
      daxpy_(&length, &alpha, da, &incx, atb, &incy);
      //LOG(INFO) << " atb:" << mit::DebugStr(atb,3, 15);
      /*
      if (cli_param_.debug) {
        LOG(INFO) << " atb after update:" << mit::DebugStr(atb, 10, 15);
      }
      */
    }
  }
  /*! \brief merge another equation*/
  void merge(NormalEquation* other) {
    CHECK(other->length == length);
    mit_double alpha = 1.0;
    mit_int incx = 1;
    mit_int incy = 1;
    daxpy_(&triK, &alpha, other->ata, &incx, ata, &incy);
    daxpy_(&length, &alpha, other->atb, &incx, atb, &incy);
  }
  /*! \brief reset left matrix ata and right vector atb*/ 
  void reset() {
    for (mit_int i = 0; i < triK; i++){
      ata[i] = 0.0;
    }
    for (mit_int i = 0; i < length; i++){
      atb[i] = 0.0;
    }
  }
  //length of left matrix array
  mit_int triK; 
  //length of latent vector
  mit_int length;
  //array of left handside matrix(upper triangle form of the symmetric matrix),  
  mit_double* ata;
  //right handside of the vector
  mit_double* atb;
  //the current lantent vector
  mit_double* da;
  //the matrix type, 'U' for upper triangle form, and 'L' for lower triangle form of the symmetric matrix
  mit_char type;
};


/*!
 * \brief optimizer: als algorithm
 */
class ALSOptimizer : public Optimizer {
  public:
    /*! \brief constructor for als */
    ALSOptimizer(const mit::KWArgs & kwargs);
    
    /*! \brief destructor */
    ~ALSOptimizer();
    
    /*! \brief get ALS optimizer */
    static ALSOptimizer * Get(const mit::KWArgs & kwargs) {
      return new ALSOptimizer(kwargs);
    }
    
    void Init(mit_uint dim) override {}

    /*!
     * \brief parameter updater for mpi
     * \param idx model index 
     * \param g gradient of model index 
     * \param w model index weight
     */
     void Update(const mit_uint idx,
                const mit_float g,
                mit_float & w) override {};
    
    /*! 
     * \brief model updater for parameter server interface
     * \param param optimizer parameter
     * \param key model feature id
     * \param idx entry data index
     * \param g gradient of unit index that computed by worker node
     * \param w model parameter of unit index 
     * \param weight used initialize optimizer middle variable
     */
    void Update(const mit::OptimizerParam & param, 
                const mit_uint & key, 
                const size_t & idx, 
                const mit_float & g,
                mit_float & w,
                mit::Entry * weight = nullptr) override {};
    
    void Update(const mit_uint & key, 
                const size_t & idx, 
                const mit_float & g, 
                mit_float & w, 
                mit::Entry * weight = nullptr) override {};

    void Update(std::unordered_map<ps::Key, mit::mit_float>& rating_map,
                std::vector<ps::Key>& user_keys,
                std::vector<mit_float> & user_weights,
                std::vector<int> & user_lens,
                std::vector<ps::Key> & item_keys,
                std::vector<mit_float> & item_weights,
                std::vector<int> & item_lens,
                std::vector<mit_float> * user_res_vector,
                std::vector<mit_float> * item_res_vector);
    /*!
     * \brief solve laten factor by cholesky factorization
     * \param ne norm equation
     * \param lambda l2 regularizaion factor
     * \param res_vector result vector
     * \param offset the res_vector offset
     */
    void CholeskySolve(NormalEquation* ne,
                       mit_float lambda,
                       std::vector<mit_float>* res_vector,
                       mit_uint offset);
    /*!
     * \brief init equation, for efficient solving implicit als
     * \param ne norm equation
     * \param keys user or item ids
     * \param weights user or item factor weights
     * \param lens user or item lengths
     */
    void initEquation(NormalEquation* ne,
                      const std::vector<ps::Key>& keys,
                      const std::vector<mit_float>& weights,
                      const std::vector<int>& lens);
  private:
    /*! \brief als parameter */
    mit::OptimizerParam param_; 
}; // class ALSOptimizer


ALSOptimizer::ALSOptimizer(const mit::KWArgs & kwargs) {
  param_.InitAllowUnknown(kwargs);
  this->param_w_.InitAllowUnknown(kwargs);
  cli_param_.InitAllowUnknown(kwargs);
  if (cli_param_.implicit) {
    LOG(INFO) << "implicit als initialization completed";
  }
  else {
    LOG(INFO) << "explicit als initialization completed";
  }
}

ALSOptimizer::~ALSOptimizer() {}

void ALSOptimizer::Update(std::unordered_map<ps::Key, mit::mit_float>& rating_map,
                          std::vector<ps::Key>& user_keys,
                          std::vector<mit_float> & user_weights,
                          std::vector<int> & user_lens,
                          std::vector<ps::Key> & item_keys,
                          std::vector<mit_float> & item_weights,
                          std::vector<int> & item_lens,
                          std::vector<mit_float> * user_res_vector,
                          std::vector<mit_float> * item_res_vector) {
  CHECK_EQ(user_keys.size(), user_lens.size());
  CHECK_EQ(user_weights.size(), user_res_vector->size());
  CHECK_EQ(item_keys.size(), item_lens.size());
  CHECK_EQ(item_weights.size(), item_res_vector->size());
  auto user_feature_size = user_keys.size();
  auto item_feature_size = item_keys.size();
  CHECK(user_lens.size() > 0);
  if (cli_param_.debug) {
    LOG(INFO) << "norm equation latent length:" << user_lens[0];
  }
  //begin solving each factor 
  NormalEquation* ne = new NormalEquation(user_lens[0]);
  NormalEquation* ne_pre = new NormalEquation(user_lens[0]);
  //optimize user latent vector
  size_t user_offset = 0;
  size_t item_offset = 0;
  mit_uint rating_num = 0;
  
  if (cli_param_.implicit) {
    initEquation(ne_pre, item_keys, item_weights, item_lens);
  }
  for (auto i = 0u; i < user_feature_size; i++){
    ne->reset();
    if (cli_param_.implicit) {
      ne->merge(ne_pre);
    }
    LOG(INFO) << "ne->ata:" << mit::DebugStr(ne->ata, 3, 10);
    LOG(INFO) << "ne->atb:" << mit::DebugStr(ne->atb, 2, 10);
    item_offset = 0;
    rating_num = 0; 
    mit_uint user_id = user_keys[i];
    mit_uint user_len = user_lens[i];
    for (auto j = 0u; j < item_feature_size; j++){
      mit_uint item_id = item_keys[j];
      mit_uint item_len = item_lens[j];
      CHECK_EQ(user_len, item_len);
      mit_uint new_key = mit::NewKey(
        user_id, item_id, cli_param_.nbit);
      if (rating_map.find(new_key) != rating_map.end()){
        if (cli_param_.implicit){
          mit_float c = param_.alpha * rating_map[new_key];
          ne->add(item_weights, item_offset, item_len, (c + 1.0) / c, c);
        }
        else {
          ne->add(item_weights,
                  item_offset,
                  item_len,
                  rating_map[new_key]);
        }
        rating_num++;
      }
      item_offset += item_len;
    }
    //get solve result of user i
    CholeskySolve(ne, rating_num * param_.l2, user_res_vector, user_offset);
    user_offset += user_len;
  }
  //solve item latent vector
  user_offset = 0;
  item_offset = 0;
  if (cli_param_.implicit) {
    initEquation(ne_pre, item_keys, *user_res_vector, item_lens);
  }
  for (auto i = 0u; i < item_feature_size; i++){
    ne->reset();
    if (cli_param_.implicit) {
      ne->merge(ne_pre);
    }
    user_offset = 0;
    rating_num = 0; 
    mit_uint item_id = item_keys[i];
    mit_uint item_len = item_lens[i];
    for (auto j = 0u; j < user_feature_size; j++){
      mit_uint user_id = user_keys[j];
      mit_uint user_len = user_lens[j];
      CHECK_EQ(user_len, item_len);
      mit_uint new_key = mit::NewKey(
        user_id, item_id, cli_param_.nbit);
      if (rating_map.find(new_key) != rating_map.end()){
        if (cli_param_.implicit){
          mit_float c = param_.alpha * rating_map[new_key];
          ne->add(*user_res_vector, user_offset, user_len, (c + 1.0) / c, c); 
        }   
        else {
          ne->add(*user_res_vector,
                  user_offset,
                  user_len,
                  rating_map[new_key]);
        }   
        rating_num++;
      }
      user_offset += user_len;
    }
    //get solve result of item i
    CholeskySolve(ne, rating_num * param_.l2, item_res_vector, item_offset);
    item_offset += item_len;
  }
  delete ne;
}

void ALSOptimizer::initEquation(NormalEquation* ne,
                                const std::vector<ps::Key>& keys,
                                const std::vector<mit_float>& weights,
                                const std::vector<int>& lens){
  CHECK_EQ(keys.size(), lens.size());
  CHECK(lens.size() > 0);
  CHECK_EQ(lens[0], ne->length);
  ne->reset();
  size_t factor_num = keys.size();
  size_t factor_len = lens[0];
  mit_uint offset = 0;
  for (size_t i = 0u; i < factor_num; i++){
    CHECK_EQ(lens[i], factor_len);
    LOG(INFO) << "ne_pre->ata:" << mit::DebugStr(ne->ata, 3, 10);
    LOG(INFO) << "ne_pre->atb:" << mit::DebugStr(ne->atb, 2, 10);
    ne->add(weights, offset, factor_len, 0);
    LOG(INFO) << "ne_pre->ata:" << mit::DebugStr(ne->ata, 3, 10);
    LOG(INFO) << "ne_pre->atb:" << mit::DebugStr(ne->atb, 2, 10);
    offset += lens[i];
  }
}


void ALSOptimizer::CholeskySolve(NormalEquation* ne, 
                                 mit_float lambda,
                                 std::vector<mit_float>* res_vector,
                                 mit_uint offset) {
  mit_int i = 0;
  mit_int j = 2;
  //add regualarization factor to the diagonals of AtA
  while (i < ne->triK) {
    ne->ata[i] += lambda;
    i += j;
    j++;
  }
  //matrix type, 'U' or 'L'
  mit_char matrix_type = ne->type;
  //factor length
  mit_int k = ne->length;
  //number of right hand sides
  mit_int nrhs = 1;
  //return value, 0 means successful exit
  mit_int info = 0;
  if (cli_param_.debug) {
    LOG(INFO) << "ata:" << mit::DebugStr(ne->ata, 3, 10);
    LOG(INFO) << "atb:" << mit::DebugStr(ne->atb, 2, 10);
  }
  //solve result factor by cholesky factorization
  dppsv_(&matrix_type, &k, &nrhs, ne->ata, ne->atb, &k, &info);
  if (cli_param_.debug) {
    LOG(INFO) << "info:" << info;
    LOG(INFO) << "solve result:" << mit::DebugStr(ne->atb, 2, 10);
  }
  
  //store in the result factor array
  if (0 == info){
    i = 0;
    while (i < k) {
      (*res_vector)[offset + i] = ne->atb[i];
      i += 1;
    }
  }
  ne->reset();
}

} // namespace mit
#endif // OPENMIT_OPTIMIZER_ALS_H_
