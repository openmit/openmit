#ifndef OPENMIT_LOSS_H_
#define OPENMIT_LOSS_H_

#include <cmath>
#include <string>
#include <vector>

#include "dmlc/data.h"
#include "dmlc/logging.h"

#include "openmit/common/base.h"

namespace mit {

/*! 
 * \brief loss function template for various loss metric 
 */
class Loss {
  public:
    /*! \brief create a loss function */
    static Loss * Create(std::string loss_type);
    
    /*! \brief destructor */
    virtual ~Loss() {}

    /*! \brief loss function */
    virtual mit_float LossFunc(
        const mit_float & pred, const mit_float & label) = 0;

    /*! \brief calculate loss based on a batch-data */
	mit_float CalcLoss(const dmlc::RowBlock<mit_uint>& row_block, 
                       const std::vector<mit_float>& pred_value);

    /*! \brief get loss type */
	inline const std::string& Type() const { return type_; }

protected:
    /*! \brief loss type */
	std::string type_;

}; // class Loss

/*!
 * \brief squared-loss function
 */
class SquaredLoss: public Loss {
  public:
    /*! \brief constructor */
    SquaredLoss() { this->type_ = "squared"; }
    
    /*! \brief destructor */
    virtual ~SquaredLoss() {}
    
    /*! \brief get square loss */
    static SquaredLoss * Get() {
      static SquaredLoss loss;
      return & loss;
    }

    /*! \brief square-loss complication */
    mit_float LossFunc(const mit_float & pred, 
                       const mit_float & label) override;
}; // class SquaredLoss 

/*! 
 * \brief logit-loss, such as lr, fm, ffm etc
 */
class LogitLoss: public Loss {
  public:
    /*! \brief constructor */
    LogitLoss() { this->type_ = "logit"; }
    
    /*! \brief destructor */
    virtual ~LogitLoss() {}
    
    /*! \brief get logit loss */
    static LogitLoss * Get() {
      static LogitLoss loss;
      return & loss;
    }

    /*! \brief logit-loss complication */
    mit_float LossFunc(const mit_float & pred, 
                       const mit_float & label) override;

}; // class LogitLoss 


} // namespace mit

#endif // OPENMIT_LOSS_H_
