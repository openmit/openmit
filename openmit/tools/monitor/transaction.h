/*!
 *  Copyright 2016 by Contributors
 *  \file transaction.h
 *  \brief monitoring program running transaction state
 *  \author ZhouYong, diffm
 */
#ifndef OPENMIT_TOOLS_MINITOR_TRANSACTION_H_
#define OPENMIT_TOOLS_MINITOR_TRANSACTION_H_

#include <stack>
#include <stdint.h>
#include "dmlc/logging.h"
#include "openmit/tools/util/timer.h"

namespace mit {
/*!
 * \brief trace transaction message
 */
struct TMessage {
  /*! \brief trace type. such as 
   *         "commincation","gradient","predict",
   *         "ps","admm","worker","epoch" etc. 
   */
  std::string type;
  /*! \brief tracke concrete info name */
  std::string name;
  /*! 
   * \breif trace info level. such as '0','1',...
   *        the smaller value, the easier it is to trace. 
   */
  uint32_t level;
  /*! \brief status timestamp */
  uint64_t timestamp;

  /*! \brief constructor */
  TMessage() : type(""), name(""), level(0), 
               timestamp(mit::TimeStamp()) {}

  TMessage(uint32_t level, 
           std::string type, 
           std::string name) :
    type(type), name(name), level(level), 
    timestamp(mit::TimeStamp()) {}

  /*! \brief message string */
  std::string MessageStr() const {
    return "<" + std::to_string(level) + ", " 
          + type + ", " + name + ">";
  }
}; // struct TMessage

/*! 
 * \brief transaction status class that be used to 
 *        record task information at the excution phase
 */
class Transaction {
  public:
    /*! \brief default constructor */
    Transaction() : message_(0, "", ""), print_(false) {}
    /*! \brief constructor by level, type, name */
    Transaction(uint32_t level, std::string type, 
                std::string name, 
                bool print = false) : 
      message_(level, type, name), print_(print) {
    if (print) 
      LOG(INFO) << "transaction " << message_.MessageStr() << " begin.";
  }

    /*! \brief destructor */
    ~Transaction() { }
    
    /*! \brief a transaction */
    static Transaction * Create(uint32_t level, 
                                std::string type, 
                                std::string name);

    /*! \brief create a transaction */
    static void Create(Transaction * trans);
    
    /*! \brief end a transaction */
    static void End(Transaction * trans);

    /*! \brief stack size */
    inline static uint32_t Size() { 
      return trans_info.size(); 
    }
    
    /*! \brief transaction level */
    inline uint32_t Level() const { 
      return message_.level; 
    }
    
    /*! \brief transaction type */
    inline std::string Type() const { 
      return message_.type; 
    }
    
    /*! \brief transaction name */
    inline std::string Name() const { 
      return message_.name; 
    }
    
    /*! \brief transaction timestamp */
    inline uint64_t TimeStamp() const {
      return message_.timestamp;
    }

    /*! \brief store transaction informzation */
    static std::stack<Transaction *> trans_info;
    /*! \brief used whether init */
    static bool _init;

  private:
    /*! \brief initialize transaction */
    static bool Init();

    /*! \brief trace log */
    void LogTrace(Transaction * trans);
  private:
    /*! \brief transaction message unit */
    TMessage message_;
    /*! \brief is print */
    bool print_;
}; // class Transaction

} // namespace mit

#endif // OPENMIT_TOOLS_MINITOR_TRANSACTION_H_
