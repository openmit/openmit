#ifndef OPENMIT_ENGINE_UPDATER_H_
#define OPENMIT_ENGINE_UPDATER_H_

#include <memory>
#include <unordered_map>

#include "dmlc/parameter.h"
#include "ps/ps.h"

#include "openmit/common/arg.h"
#include "openmit/entity/unit.h"
#include "openmit/optimizer/optimizer.h"

namespace mit {

/*!
 * \brief optimizer related parameter
 */
class UpdaterParam : public dmlc::Parameter<UpdaterParam> {
  public:
    /*! \brief optimizer type */
    std::string optimizer;
    /*! \brief number of field */
    mit_uint field_num;
    /*! \brief latent factor length */
    mit_uint k;

    /*! \brief declare parameters */
    DMLC_DECLARE_PARAMETER(UpdaterParam) {
      DMLC_DECLARE_FIELD(optimizer).set_default("ftrl");
      DMLC_DECLARE_FIELD(field_num).set_default(0);
      DMLC_DECLARE_FIELD(k).set_default(0);
    }
}; // class UpdaterParam

/*!
 * \brief 
 */
class Updater {
  public:
    Updater(const mit::KWArgs & kwargs);
    ~Updater();

    void Init(const mit::KWArgs & kwargs);

    void Run(const ps::KVPairs<mit_float> * req_data,
             PMAPT * weight);
    
    std::string Name() { return param_.optimizer; }

    UpdaterParam Param() const { return param_; }
  
  private:
    void Update(PMAPT & map_grad, PMAPT * weight);

  private:
    /*! \brief optimizer */
    std::shared_ptr<mit::Opt> opt_;

    /*! \brief optimizer parameter */
    UpdaterParam param_;

}; // class Updater

} // namespace mit

#endif // OPENMIT_ENGINE_UPDATER_H_
