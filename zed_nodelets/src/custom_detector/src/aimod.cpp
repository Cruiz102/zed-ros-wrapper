#include "aimod.hpp"

namespace zed_nodelets {

/**
 * Convert Zed images to OpenCV images
 *
 * @param input Zed image to be converted
 * @return OpenCV image virtually identical to the Zed one
 */
cv::Mat zedMat2cvMat(sl::Mat input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

AI::AI(std::string input_yaml) {

    
    // Check if the directory is a yaml document.
    if(std::filesystem::path(input_yaml) != "yaml"){
        std::cout << "File extension is not yaml." std::endl;
    }
    // Load yolov7
    // The YAML::LoadFile function is already imported when using the Yolov7 detector.
    YAML::Node root = YAML::LoadFile(input_yaml);
    Yolov7 yolov7 = new Yolov7(root["yolov7"]);
    yolov7.LoadEngine();
}
// All function that return DetectedObjects now will return CustomobjectData
sl::CustomBoxObjectData AI::detect_objects(std::vector<cv::Mat> &frames) {
    //
    sl::CustomBoxObjectData results;
    std::vector<sl::CustomBoxObjectData> objects_in;

    //Predict items from the frames
    std::vector<DetectRes> predictions = yolov7->InferenceImages(frames);
    for(Bbox &prediction : predictions.det_results) {
        cv::Rect bounding_box = cv::Rect(prediction.x, prediction.y, prediction.w, prediction.h);

        //Create the CustomBoxObjectData
        sl::CustomBoxObjectData tmp;

        //Setting the Bounding box info
        std::vector<sl::uint2> bbox_out(4);
        bbox_out[0] = sl::uint2(prediction.x, prediction.y);
        bbox_out[1] = sl::uint2(prediction.x + predition.width, prediction.y);
        bbox_out[2] = sl::uint2(prediction.x + prediction.width, prediction.y + prediction.height);
        bbox_out[3] = sl::uint2(prediction.x, prediction.y + prediction.height);


        // Fill the detections into the correct SDK format
        tmp.unique_object_id = sl::generate_unique_id();
        tmp.probability = prediction.prop;
        tmp.label = (int) prediction.class;
        tmp.bounding_box_2d = bbox_out;
        tmp.is_grounded = true; // objects are moving on the floor plane and tracked in 2D only
        objects_in.push_back(tmp);

    return objects_in;
    }
}



 sl::CustomBoxObjectData AI::detect(<sl::Mat> &image) {
    // Inference need to be a Vector type. So we are creating
    // a vector with a single element.
    auto vect = std::vector<cv::Mat>;
    vect.push_back(zedMat2cvMat(image));
    return this->detect_objects(vect);
}
    
    } // namespace zed_nodelets