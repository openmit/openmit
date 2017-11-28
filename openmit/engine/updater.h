/*!
 * Copyright 2016 by Contributors
 * \brief model parameter updater
 * \author ZhouYong
 */
#ifndef OPENMIT_ENGINE_UPDATER_H_
#define OPENMIT_ENGINE_UPDATER_H_ 

namespace mit {
class Updater {
public:
  explicit Updater(const mit::KWArgs & kwargs);
  virtual ~Updater() {}
} // namespace mit

#endif // OPENMIT_ENGINE_UPDATE_H_
