#ifndef OPENMIT_OPTIMIZER_GD_H_
#define OPENMIT_OPTIMIZER_GD_H_

#include "openmit/optimizer/optimizer.h"

namespace mit {

/*! 
 * \brief gradient descent parameter
 */
class GDParam : public dmlc::Parameter<GDParam> {
  public:
    /*! \brief optimizer type. gd/adagrad/... */
    std::string optimizer_type;

    /*! \brief \alpha learning rate */
    float alpha;
    /*! 
     * \brief whether or not use learning rate parameter 
     *        gd / adagrad
     */
    bool is_alpha;
    /*! \brief l1 regularation */
    float l1;
    /*! \brief l2 regularation */
    float l2;
  
    /*! \brief declare parameters */
    DMLC_DECLARE_PARAMETER(GDParam) {
      DMLC_DECLARE_FIELD(optimizer_type).set_default("gd");
      DMLC_DECLARE_FIELD(alpha).set_default(0.01);
      DMLC_DECLARE_FIELD(is_alpha).set_default(true);
      DMLC_DECLARE_FIELD(l1).set_default(0.1);
      DMLC_DECLARE_FIELD(l2).set_default(0.1);
    }

}; // class GDParam

/*!
 * \brief optimizer: gradient descent algorithm
 *        support: sgd/batch-gd
 */
class GD : public Opt {
  public:
    /*! \brief constructor for GD */
    GD(const mit::KWArgs & kwargs);
    /*! \brief destructor */
    ~GD();
    /*! \brief get GD optimizer */
    static GD * Get(const mit::KWArgs & kwargs) {
      return new GD(kwargs);
    }
    /*! \brief parameter updater for mpi */
    void Update(
        const dmlc::Row<mit_uint> & row, 
        mit_float pred, 
        mit::SArray<mit_float> & weight_) override;
    /*! \brief parameter updater for ps */
    inline void Update(
        std::unordered_map<ps::Key, mit::Unit * > & map_grad,
        std::unordered_map<ps::Key, mit::Unit * > * weight) override;

  private:
    /*! \brief gradient algorithm parameter */
    GDParam param_;
    /*! \brief n gradient squared sum for adaptive gd for ps */
    std::unordered_map<mit_uint, mit::Unit * > nm_;
    /*! \brief n squared-sum weight for each features for mpi */
    mit::SArray<mit_float> nv_;

}; // class GD

} // namespace mit

#endif // OPENMIT_OPTIMIZER_GD_H_
