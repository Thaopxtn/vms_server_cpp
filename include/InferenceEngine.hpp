// InferenceEngine.hpp - C++ Header cho module suy luận AI (YOLO & SFace)
#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Cấu trúc kết quả phát hiện đối tượng (Bounding Box, Confidence, Class)
struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

struct OnnxModelInfo {
    bool loaded = false;
    bool detector_compatible = false;
    bool face_embedding_compatible = false;
    std::string description;
    std::string error;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
};

OnnxModelInfo inspectOnnxModel(const std::string& model_path);

// Lớp suy luận YOLOv11 sử dụng ONNX Runtime
class ONNXDetector {
public:
    ONNXDetector();
    ~ONNXDetector() = default;

    // Tải mô hình YOLO ONNX
    bool loadModel(const std::string& model_path, bool use_cuda = false);
    const std::string& getLastError() const { return last_error; }

    // Phát hiện đối tượng trong frame ảnh
    std::vector<Detection> detect(const cv::Mat& frame, float conf_threshold = 0.4f, float iou_threshold = 0.45f);

private:
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions session_opts;

    // Chi tiết Model Input/Output
    std::vector<int64_t> input_shape;
    std::string input_name;
    std::vector<std::string> output_names;
    std::vector<int64_t> output_shape;
    std::string last_error;
    bool is_plate_segment_model = false;
    
    int num_classes = 80; // Mặc định YOLOv11 COCO
};

// Lớp trích xuất đặc trưng khuôn mặt sử dụng mô hình SFace ONNX
class FaceEmbedder {
public:
    FaceEmbedder();
    ~FaceEmbedder() = default;

    // Tải mô hình SFace
    bool loadModel(const std::string& model_path, bool use_cuda = false);
    const std::string& getLastError() const { return last_error; }

    // Trích xuất vector đặc trưng 128-d từ ảnh khuôn mặt (kích thước chuẩn 112x112)
    std::vector<float> extractEmbedding(const cv::Mat& face_crop);

    // Tính toán Cosine Similarity giữa 2 vector khuôn mặt
    static float calculateSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2);

private:
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions session_opts;

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    std::string input_name;
    std::vector<std::string> output_names;
    std::string last_error;
};

#endif // INFERENCE_ENGINE_HPP
