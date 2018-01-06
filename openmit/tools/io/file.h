#include <dirent.h> // rmdir
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>  // access

namespace mit {

class DirFile {
  public:
    /*! \brief check wether dir_has exist. failure: -1 */
    static bool access_dir(const char* path) {
      if (!path) return -1;
      return access(path, 0) == 0;
    }

    /*! \brief mdkir a dir. success: 0 */
    static bool mk_dir(const char* path) {
      if (!path) return -1;
      return mkdir(path, 0777) == 0;
    };

    /*! \brief delete a dir */
    static bool rm_dir(const char* path) {
      if (!path) return -1;
      return rmdir(path) == 0;
    }
}; // class File

} // namespace mit
