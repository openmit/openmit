/*!
<<<<<<< HEAD
 *  Copyright 2016 by Contributors
=======
 *  Copyright (c) 2017 by Contributors
>>>>>>> ps
 *  \file ffm.h
 *  \brief field-aware factorization machine model
 *  \author ZhouYong, diffm
 */
<<<<<<< HEAD
#ifndef OPENMIT_MODELS_FFM_H_
#define OPENMIT_MODELS_FFM_H_

#include "openmit/model/model.h"

namespace mit {
/*!
 * \brief the field-aware factorization machine model
 */
class FFM : public Model {
public:
  /*! \brief default constructor */
  FFM(const mit::KWArgs & kwargs) : Model(kwargs) {
    optimizer_v_.reset(
      mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
  }

  /*! \brief destructor */
  virtual ~FFM() {};

  /*! \brief get ffm-model pointer */
  static FFM * Get(const mit::KWArgs & kwargs) { 
    return new FFM(kwargs); 
  }

  public:  // method for server
    /*! \brief pull request */
    void Pull(ps::KVPairs<mit_float> & response, 
              mit::entry_map_type * weight) override;
 
    /*! \brief update */
    void Update(const ps::SArray<mit_uint> & keys, 
                const ps::SArray<mit_float> & vals, 
                const ps::SArray<int> & lens, 
                mit::entry_map_type * weight) override;

  public:
    /*! \brief calcuate gradient based on one instance for ps */
=======
#ifndef OPENMIT_MODEL_FFM_H_
#define OPENMIT_MODEL_FFM_H_

#include "openmit/model/model.h"
#include "openmit/model/psmodel.h"

namespace mit {
/*!
 * \brief the field-aware factorization machine model 
 *        that be suitable for mpi or local
 */
class FFM : public Model {
  public:
    /*! \brief constructor */
    FFM(const mit::KWArgs& kwargs) : Model(kwargs) {
      optimizer_v_.reset(
        mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
    }

    /*! \brief destructor */
    virtual ~FFM();

    /*! \brief get ffm-model pointer */
    static FFM* Get(const mit::KWArgs& kwargs);

    /*! \brief calculate model gradient based one instance */
    void Gradient(const dmlc::Row<mit_uint>& row,
                  const mit_float& pred,
                  mit::SArray<mit_float>* grad) override;

    /*! \brief prediction based one instance */
    mit_float Predict(const dmlc::Row<mit_uint>& row,
                      const mit::SArray<mit_float>& weight,
                      bool norm) override;

  private:
    /*! \brief lr model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
}; // class FFM 

/*!
 * \brief the field-aware factorization machine model 
 *        that be suitable for ps framework
 */
class PSFFM : public PSModel {
  public:
    /*! 
     * \brief default constructor. 
     *        initialize custom optimizer for embedding learning
     */
    PSFFM(const mit::KWArgs& kwargs) : PSModel(kwargs) {
      optimizer_v_.reset(
        mit::Optimizer::Create(kwargs, cli_param_.optimizer_v));
    }

    /*! \brief destructor */
    virtual ~PSFFM();

    /*! \brief get ffm-model pointer */
    static PSFFM* Get(const mit::KWArgs& kwargs);

    /*! \brief pull request method for server */
    void Pull(ps::KVPairs<mit_float> & response, 
              mit::entry_map_type * weight) override;
 
    /*! \brief calcuate gradient based on one instance */
>>>>>>> ps
    void Gradient(const dmlc::Row<mit_uint> & row, 
                  const std::vector<mit_float> & weights,
                  mit::key2offset_type & key2offset,
                  std::vector<mit_float> * grads,
                  const mit_float & lossgrad_value) override; 

<<<<<<< HEAD
    /*! \brief calculate model gradient based one instance for mpi */
    void Gradient(const dmlc::Row<mit_uint> & row,
                  const mit_float & pred,
                  mit::SArray<mit_float> * grad) override;

  public:
    /*! \brief prediction based one instance for ps */
=======
    /*! \brief prediction based one instance */
>>>>>>> ps
    mit_float Predict(const dmlc::Row<mit_uint> & row, 
                      const std::vector<mit_float> & weights, 
                      mit::key2offset_type & key2offset, 
                      bool is_norm) override;

<<<<<<< HEAD
    /*! \brief prediction based one instance for mpi */
    mit_float Predict(const dmlc::Row<mit_uint> & row,
                      const mit::SArray<mit_float> & weight,
                      bool is_norm) override;
=======
    /*! \brief update */
    void Update(const ps::SArray<mit_uint> & keys, 
                const ps::SArray<mit_float> & vals, 
                const ps::SArray<int> & lens, 
                mit::entry_map_type * weight) override;
>>>>>>> ps

  private:
    /*! \brief ffm 1-order linear item */
    mit_float Linear(const dmlc::Row<mit_uint> & row, 
                     const std::vector<mit_float> & weights, 
                     mit::key2offset_type & key2offset);

    /*! \brief ffm 2-order cross item */
    mit_float Cross(const dmlc::Row<mit_uint> & row, 
                    const std::vector<mit_float> & weights, 
                    mit::key2offset_type & key2offset);

  private:
    /*! \brief lr model optimizer for v */
    std::unique_ptr<mit::Optimizer> optimizer_v_;
<<<<<<< HEAD
}; // class FFM 

} // namespace mit
#endif // OPENMIT_MODELS_FFM_H_
=======
}; // class PSFFM 

} // namespace mit
#endif // OPENMIT_MODEL_FFM_H_
>>>>>>> ps
