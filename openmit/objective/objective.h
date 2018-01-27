#ifndef OPENMIT_OBJECTIVE_OBJECTIVE_H_
#define OPENMIT_OBJECTIVE_OBJECTIVE_H_ 

namespace mit {
/*!
 * \brief objective function of machine learning
 */
class ObjFunction {
public:
  ObjFunction() {}
  ~ObjFunction() {}

  using ObjFunc = std::function<float(float &, float &)>;
  
private:
  ObjFunc obj_func_;
  ObjFunc gradient_;
}; // class ObjFunction
} // namespace mit

#endif // OPENMIT_OBJECTIVE_OBJECTIVE_H_
