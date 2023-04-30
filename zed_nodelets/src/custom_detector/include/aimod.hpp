#ifndef __AI_MODULE__
#define __AI_MODULE__
// #include <filesystem>
#include <experimental/filesystem>
// We have to include the Zed Camera
#include <sl/Camera.hpp>

// This is the file from the Detector of Yolov7
#include "yolov7.h"


// This variable is used for storing the CustomDetectedObjects that
// is transfer to the zed.ingestinference(). This vector is updated
// every grab of the camera.
using CustomDetectedObjects = std::vector<CustomBoxObjectData>;

namespace zed_nodelets{
class AI {
    private:
        Yolov7* yolov7;
public:
        AI(std::string input_yaml);
        CustomBoxObjectData detect_objects(sl::Mat &frame); //Detects the object
        };

}
#endif