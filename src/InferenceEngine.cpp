// InferenceEngine.cpp - C++ Source cho module suy luận AI (YOLO & SFace)
#include "InferenceEngine.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

class ScopedStderrSilencer {
public:
    ScopedStderrSilencer() {
        std::fflush(stderr);
#ifdef _WIN32
        saved_fd_ = _dup(_fileno(stderr));
        if (saved_fd_ != -1) {
            FILE* ignored = nullptr;
            active_ = freopen_s(&ignored, "NUL", "w", stderr) == 0;
        }
#else
        saved_fd_ = dup(fileno(stderr));
        null_fd_ = open("/dev/null", O_WRONLY);
        if (saved_fd_ != -1 && null_fd_ != -1) {
            active_ = dup2(null_fd_, fileno(stderr)) != -1;
        }
#endif
    }

    ScopedStderrSilencer(const ScopedStderrSilencer&) = delete;
    ScopedStderrSilencer& operator=(const ScopedStderrSilencer&) = delete;

    ~ScopedStderrSilencer() {
        std::fflush(stderr);
#ifdef _WIN32
        if (active_ && saved_fd_ != -1) {
            _dup2(saved_fd_, _fileno(stderr));
        }
        if (saved_fd_ != -1) _close(saved_fd_);
#else
        if (active_ && saved_fd_ != -1) {
            dup2(saved_fd_, fileno(stderr));
        }
        if (null_fd_ != -1) close(null_fd_);
        if (saved_fd_ != -1) close(saved_fd_);
#endif
    }

private:
    bool active_ = false;
    int saved_fd_ = -1;
#ifndef _WIN32
    int null_fd_ = -1;
#endif
};

struct YoloOutputLayout {
    int attrs = 0;
    int predictions = 0;
    bool transposed = false;
};

std::string shapeToString(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ",";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

bool isNchwImageInput(const std::vector<int64_t>& shape, std::string& error) {
    if (shape.size() != 4) {
        error = "input rank " + std::to_string(shape.size()) + " khong phai NCHW 4D";
        return false;
    }
    if (shape[1] != 3) {
        error = "input shape " + shapeToString(shape) + " khong phai NCHW voi 3 kenh mau";
        return false;
    }
    if (shape[2] == 0 || shape[3] == 0) {
        error = "input H/W khong hop le: " + shapeToString(shape);
        return false;
    }
    return true;
}

bool inferYoloOutputLayout(const std::vector<int64_t>& shape, YoloOutputLayout& layout, std::string& error) {
    if (shape.size() != 3) {
        error = "output rank " + std::to_string(shape.size()) + " khong phai YOLO 3D";
        return false;
    }

    const int64_t dim1 = shape[1];
    const int64_t dim2 = shape[2];
    if (dim1 >= 5 && (dim2 < 0 || dim1 <= dim2)) {
        layout.attrs = static_cast<int>(dim1);
        layout.predictions = dim2 > 0 ? static_cast<int>(dim2) : 0;
        layout.transposed = false;
        return true;
    }
    if (dim2 >= 5) {
        layout.attrs = static_cast<int>(dim2);
        layout.predictions = dim1 > 0 ? static_cast<int>(dim1) : 0;
        layout.transposed = true;
        return true;
    }

    error = "output shape " + shapeToString(shape) + " khong giong YOLO [1,attrs,boxes] hoac [1,boxes,attrs]";
    return false;
}

bool isYoloDetectorModel(const std::vector<int64_t>& input_shape,
                         const std::vector<int64_t>& output_shape,
                         std::string& error) {
    if (!isNchwImageInput(input_shape, error)) return false;
    YoloOutputLayout layout;
    if (!inferYoloOutputLayout(output_shape, layout, error)) return false;
    if (layout.attrs < 5) {
        error = "YOLO output can toi thieu 5 thuoc tinh moi box";
        return false;
    }
    return true;
}

bool isFaceEmbeddingModel(const std::vector<int64_t>& input_shape,
                          const std::vector<int64_t>& output_shape,
                          std::string& error) {
    if (!isNchwImageInput(input_shape, error)) return false;
    if (input_shape[2] > 0 && input_shape[2] != 112) {
        error = "embedding input H phai la 112, dang la " + std::to_string(input_shape[2]);
        return false;
    }
    if (input_shape[3] > 0 && input_shape[3] != 112) {
        error = "embedding input W phai la 112, dang la " + std::to_string(input_shape[3]);
        return false;
    }

    bool has128Dim = false;
    int64_t knownElements = 1;
    bool allKnown = true;
    for (int64_t dim : output_shape) {
        if (dim == 128) has128Dim = true;
        if (dim <= 0) {
            allKnown = false;
        } else {
            knownElements *= dim;
        }
    }
    if (!has128Dim && (!allKnown || knownElements < 128)) {
        error = "embedding output khong co vector 128 chieu: " + shapeToString(output_shape);
        return false;
    }
    return true;
}

float intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect inter = a & b;
    const int inter_area = inter.area();
    if (inter_area <= 0) return 0.0f;
    const int union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0) return 0.0f;
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

std::vector<int> applyNms(const std::vector<cv::Rect>& boxes,
                          const std::vector<float>& confidences,
                          float iou_threshold) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return confidences[lhs] > confidences[rhs];
    });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        const int idx = order[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        for (size_t j = i + 1; j < order.size(); ++j) {
            const int next = order[j];
            if (!suppressed[next] && intersectionOverUnion(boxes[idx], boxes[next]) > iou_threshold) {
                suppressed[next] = true;
            }
        }
    }
    return keep;
}
Ort::Env& ortEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "VMSInference");
    return env;
}

std::vector<int64_t> firstInputShape(Ort::Session& session) {
    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    return type_info.GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<int64_t> firstOutputShape(Ort::Session& session) {
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    return type_info.GetTensorTypeAndShapeInfo().GetShape();
}

void normalizeDetectorInputShape(std::vector<int64_t>& shape) {
    if (shape.size() != 4) return;
    if (shape[0] == -1) shape[0] = 1;
    if (shape[2] == -1) shape[2] = 640;
    if (shape[3] == -1) shape[3] = 640;
}

void normalizeEmbeddingInputShape(std::vector<int64_t>& shape) {
    if (shape.size() != 4) return;
    if (shape[0] == -1) shape[0] = 1;
    if (shape[2] == -1) shape[2] = 112;
    if (shape[3] == -1) shape[3] = 112;
}

} // namespace

OnnxModelInfo inspectOnnxModel(const std::string& model_path) {
    OnnxModelInfo info;
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetLogSeverityLevel(3);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

#ifdef _WIN32
        std::wstring wide_model_path = std::wstring(model_path.begin(), model_path.end());
        ScopedStderrSilencer ort_stderr_silencer;
        Ort::Session session(ortEnv(), wide_model_path.c_str(), opts);
#else
        ScopedStderrSilencer ort_stderr_silencer;
        Ort::Session session(ortEnv(), model_path.c_str(), opts);
#endif
        if (session.GetInputCount() == 0 || session.GetOutputCount() == 0) {
            info.error = "model khong co input/output";
            return info;
        }

        info.loaded = true;
        info.input_shape = firstInputShape(session);
        info.output_shape = firstOutputShape(session);

        std::vector<int64_t> detector_input = info.input_shape;
        std::vector<int64_t> embedder_input = info.input_shape;
        normalizeDetectorInputShape(detector_input);
        normalizeEmbeddingInputShape(embedder_input);

        std::string detector_error;
        std::string embedder_error;
        info.detector_compatible = isYoloDetectorModel(detector_input, info.output_shape, detector_error);
        info.face_embedding_compatible = isFaceEmbeddingModel(embedder_input, info.output_shape, embedder_error);
        info.description = "input " + shapeToString(info.input_shape) + ", output " + shapeToString(info.output_shape);
        if (!info.detector_compatible && !info.face_embedding_compatible) {
            info.error = detector_error.empty() ? embedder_error : detector_error;
        }
    } catch (const std::exception& e) {
        info.error = e.what();
    }
    return info;
}

ONNXDetector::ONNXDetector() = default;

bool ONNXDetector::loadModel(const std::string& model_path, bool use_cuda) {
    try {
        last_error.clear();
        output_names.clear();
        output_shape.clear();
        is_plate_segment_model = false;
        session_opts = Ort::SessionOptions();
        
        if (use_cuda) {
            // Tích hợp CUDA Execution Provider nếu chạy GPU
            #ifdef ORT_API_VERSION
            OrtCUDAProviderOptions options;
            session_opts.AppendExecutionProvider_CUDA(options);
            #endif
        }

        // Tối ưu hóa luồng CPU
        session_opts.SetIntraOpNumThreads(2);
        session_opts.SetLogSeverityLevel(3);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring wide_model_path = std::wstring(model_path.begin(), model_path.end());
        {
            ScopedStderrSilencer ort_stderr_silencer;
            session = std::make_unique<Ort::Session>(ortEnv(), wide_model_path.c_str(), session_opts);
        }
#else
        {
            ScopedStderrSilencer ort_stderr_silencer;
            session = std::make_unique<Ort::Session>(ortEnv(), model_path.c_str(), session_opts);
        }
#endif

        // Lấy thông tin Input
        Ort::TypeInfo type_info = session->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shape = tensor_info.GetShape();

        // Đọc tên Input/Output từ model
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Đoạn code tương thích ngược với các phiên bản ONNX Runtime API khác nhau
        #if ORT_API_VERSION >= 12
        Ort::AllocatedStringPtr input_name_ptr = session->GetInputNameAllocated(0, allocator);
        input_name = input_name_ptr.get();
        
        Ort::AllocatedStringPtr output_name_ptr = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name_ptr.get());
        #else
        input_name = session->GetInputName(0, allocator);
        output_names.push_back(session->GetOutputName(0, allocator));
        #endif

        output_shape = firstOutputShape(*session);
        YoloOutputLayout output_layout;
        std::string layout_error;
        if (!inferYoloOutputLayout(output_shape, output_layout, layout_error)) {
            last_error = layout_error;
            session.reset();
            std::cerr << "[AI ENGINE ERROR] Model khong tuong thich detector: " << last_error << std::endl;
            return false;
        }

        // cameraAI3 exports best_plateSegment with NMS enabled as
        // [batch, 300, 38] plus a second mask-prototype output. The original
        // The cameraAI3 model is dynamic and runs much faster at 640x640.
        is_plate_segment_model = session->GetOutputCount() > 1 &&
                                 output_layout.predictions == 300 &&
                                 output_layout.attrs == 38;
        if (is_plate_segment_model) {
            if (input_shape[0] == -1) input_shape[0] = 1;
            if (input_shape[2] == -1) input_shape[2] = 640;
            if (input_shape[3] == -1) input_shape[3] = 640;
        } else {
            normalizeDetectorInputShape(input_shape);
        }

        std::string compatibility_error;
        if (!isYoloDetectorModel(input_shape, output_shape, compatibility_error)) {
            last_error = compatibility_error;
            session.reset();
            std::cerr << "[AI ENGINE ERROR] Model khong tuong thich detector: " << last_error << std::endl;
            return false;
        }

        std::cout << "[AI ENGINE] Da tai YOLO model: " << model_path 
                  << " | Input: " << input_name << " (" << input_shape[2] << "x" << input_shape[3] << ")"
                  << " | Output: " << shapeToString(output_shape)
                  << (is_plate_segment_model ? " | cameraAI3 plate segment" : "") << std::endl;
        return true;
    } catch (const std::exception& e) {
        session.reset();
        last_error = e.what();
        std::cerr << "[AI ENGINE ERROR] Khong the tai YOLO model: " << last_error << std::endl;
        return false;
    }
}

std::vector<Detection> ONNXDetector::detect(const cv::Mat& frame, float conf_threshold, float iou_threshold) {
    std::vector<Detection> detections;
    if (!session || frame.empty()) return detections;

    int img_w = frame.cols;
    int img_h = frame.rows;
    int net_w = static_cast<int>(input_shape[3]);
    int net_h = static_cast<int>(input_shape[2]);

    // 1. Tiền xử lý ảnh (Letterbox / Resize keep aspect ratio)
    float scale = std::min(static_cast<float>(net_w) / img_w, static_cast<float>(net_h) / img_h);
    int new_w = static_cast<int>(std::round(img_w * scale));
    int new_h = static_cast<int>(std::round(img_h * scale));
    
    cv::Mat resized_img;
    cv::resize(frame, resized_img, cv::Size(new_w, new_h));
    
    cv::Mat padded_img(net_h, net_w, CV_8UC3, cv::Scalar(114, 114, 114));
    int pad_left = (net_w - new_w) / 2;
    int pad_top = (net_h - new_h) / 2;
    resized_img.copyTo(padded_img(cv::Rect(pad_left, pad_top, new_w, new_h)));
    
    // Ultralytics models are trained/exported with RGB input, while OpenCV
    // frames are BGR. Keeping BGR here materially reduces detector accuracy.
    cv::Mat rgb_img;
    cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);
    cv::Mat blob;
    rgb_img.convertTo(blob, CV_32FC3, 1.0 / 255.0);

    // Chuyển đổi định dạng ảnh OpenCV HWC sang CHW (ONNX format)
    std::vector<float> input_tensor_values(net_w * net_h * 3);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(net_h, net_w, CV_32FC1, &input_tensor_values[i * net_h * net_w]);
    }
    cv::split(blob, channels);

    // 2. Tạo Ort Memory Info và Ort Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
    );

    // 3. Thực thi Suy luận (Inference)
    const char* input_names_char[] = { input_name.c_str() };
    std::vector<const char*> output_names_char;
    for (const auto& name : output_names) {
        output_names_char.push_back(name.c_str());
    }

    try {
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr}, 
            input_names_char, &input_tensor, 1, 
            output_names_char.data(), output_names_char.size()
        );

        // 4. Hậu xử lý kết quả (Postprocessing)
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto runtime_output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        // Phân tích tensor đầu ra của YOLOv11 [1, 84, 8400] hoặc [1, 5, 8400] (cho face/plate)
        YoloOutputLayout layout;
        std::string layout_error;
        if (!inferYoloOutputLayout(runtime_output_shape, layout, layout_error)) {
            std::cerr << "[AI INFERENCE ERROR] Output model khong ho tro: " << layout_error << std::endl;
            return detections;
        }
        int rows = layout.attrs;
        int cols = layout.predictions;
        if (rows < 5 || cols <= 0) return detections;

        auto valueAt = [&](int attr, int prediction) -> float {
            if (layout.transposed) {
                return output_data[prediction * rows + attr];
            }
            return output_data[attr * cols + prediction];
        };

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        float inv_scale = 1.0f / scale;

        for (int c = 0; c < cols; ++c) {
            float max_score = 0.0f;
            int class_id = -1;

            if (is_plate_segment_model) { // [x, y, w, h, conf, class_id, mask(32)]
                max_score = valueAt(4, c);
                class_id = static_cast<int>(valueAt(5, c));
            } else if (rows > 4) { // Có lớp phân loại (YOLO chuẩn)
                for (int r = 4; r < rows; ++r) {
                    float score = valueAt(r, c);
                    if (score > max_score) {
                        max_score = score;
                        class_id = r - 4;
                    }
                }
            } else { // Không có phân lớp (chỉ có 1 đối tượng duy nhất)
                max_score = valueAt(4, c);
                class_id = 0;
            }

            if (max_score >= conf_threshold) {
                int left = 0;
                int top = 0;
                int width = 0;
                int height = 0;
                if (is_plate_segment_model) {
                    // The NMS-enabled cameraAI3 export returns xyxy coordinates.
                    left = static_cast<int>((valueAt(0, c) - pad_left) * inv_scale);
                    top = static_cast<int>((valueAt(1, c) - pad_top) * inv_scale);
                    const int right = static_cast<int>((valueAt(2, c) - pad_left) * inv_scale);
                    const int bottom = static_cast<int>((valueAt(3, c) - pad_top) * inv_scale);
                    width = right - left;
                    height = bottom - top;
                } else {
                    const float cx_orig = (valueAt(0, c) - pad_left) * inv_scale;
                    const float cy_orig = (valueAt(1, c) - pad_top) * inv_scale;
                    const float w_orig = valueAt(2, c) * inv_scale;
                    const float h_orig = valueAt(3, c) * inv_scale;
                    left = static_cast<int>(cx_orig - w_orig / 2.0f);
                    top = static_cast<int>(cy_orig - h_orig / 2.0f);
                    width = static_cast<int>(w_orig);
                    height = static_cast<int>(h_orig);
                }

                const cv::Rect clipped_box = cv::Rect(left, top, width, height) &
                                             cv::Rect(0, 0, img_w, img_h);
                if (clipped_box.empty()) continue;
                boxes.push_back(clipped_box);
                confidences.push_back(max_score);
                class_ids.push_back(class_id);
            }
        }

        // Áp dụng NMS (Non-Maximum Suppression) để loại bỏ box trùng lặp
        std::vector<int> indices = applyNms(boxes, confidences, iou_threshold);

        for (int idx : indices) {
            Detection det;
            det.box = boxes[idx];
            det.confidence = confidences[idx];
            det.class_id = class_ids[idx];
            detections.push_back(det);
        }
    } catch (const std::exception& e) {
        std::cerr << "[AI INFERENCE ERROR] Loi khi suy luan: " << e.what() << std::endl;
    }

    return detections;
}

// ─── FACE EMBEDDER (SFACE IMPLEMENTATION) ───

FaceEmbedder::FaceEmbedder() = default;

bool FaceEmbedder::loadModel(const std::string& model_path, bool use_cuda) {
    try {
        last_error.clear();
        output_names.clear();
        output_shape.clear();
        session_opts = Ort::SessionOptions();
        if (use_cuda) {
            #ifdef ORT_API_VERSION
            OrtCUDAProviderOptions options;
            session_opts.AppendExecutionProvider_CUDA(options);
            #endif
        }
        session_opts.SetIntraOpNumThreads(1);
        session_opts.SetLogSeverityLevel(3);
#ifdef _WIN32
        std::wstring wide_model_path = std::wstring(model_path.begin(), model_path.end());
        {
            ScopedStderrSilencer ort_stderr_silencer;
            session = std::make_unique<Ort::Session>(ortEnv(), wide_model_path.c_str(), session_opts);
        }
#else
        {
            ScopedStderrSilencer ort_stderr_silencer;
            session = std::make_unique<Ort::Session>(ortEnv(), model_path.c_str(), session_opts);
        }
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        #if ORT_API_VERSION >= 12
        Ort::AllocatedStringPtr input_name_ptr = session->GetInputNameAllocated(0, allocator);
        input_name = input_name_ptr.get();
        
        Ort::AllocatedStringPtr output_name_ptr = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name_ptr.get());
        #else
        input_name = session->GetInputName(0, allocator);
        output_names.push_back(session->GetOutputName(0, allocator));
        #endif

        input_shape = firstInputShape(*session);
        normalizeEmbeddingInputShape(input_shape);
        output_shape = firstOutputShape(*session);

        std::string compatibility_error;
        if (!isFaceEmbeddingModel(input_shape, output_shape, compatibility_error)) {
            last_error = compatibility_error;
            session.reset();
            std::cerr << "[AI ENGINE ERROR] Model khong tuong thich embedding: " << last_error << std::endl;
            return false;
        }

        std::cout << "[AI ENGINE] Da tai SFace model: " << model_path
                  << " | Input: " << shapeToString(input_shape)
                  << " | Output: " << shapeToString(output_shape) << std::endl;
        return true;
    } catch (const std::exception& e) {
        session.reset();
        last_error = e.what();
        std::cerr << "[AI ENGINE ERROR] Khong the tai SFace model: " << last_error << std::endl;
        return false;
    }
}

std::vector<float> FaceEmbedder::extractEmbedding(const cv::Mat& face_crop) {
    std::vector<float> embedding(128, 0.0f);
    if (!session || face_crop.empty()) return embedding;

    // 1. Tiền xử lý ảnh (Resize 112x112, SFace sử dụng màu BGR thô không cần normalize 255.0)
    cv::Mat resized_face;
    cv::resize(face_crop, resized_face, cv::Size(112, 112));
    
    cv::Mat float_face;
    resized_face.convertTo(float_face, CV_32FC3);

    // Chuyển đổi HWC sang CHW
    std::vector<float> input_tensor_values(112 * 112 * 3);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(112, 112, CV_32FC1, &input_tensor_values[i * 112 * 112]);
    }
    cv::split(float_face, channels);

    // 2. Tạo Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
    );

    // 3. Thực thi suy luận trích xuất đặc trưng
    const char* input_names_char[] = { input_name.c_str() };
    std::vector<const char*> output_names_char;
    for (const auto& name : output_names) {
        output_names_char.push_back(name.c_str());
    }

    try {
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr}, 
            input_names_char, &input_tensor, 1, 
            output_names_char.data(), output_names_char.size()
        );

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        // Sao chép vector đặc trưng 128 chiều
        for (int i = 0; i < 128; ++i) {
            embedding[i] = output_data[i];
        }

        // L2 Normalize vector đặc trưng (SFace yêu cầu chuẩn hóa trước khi tính similarity)
        float l2_norm = 0.0f;
        for (float val : embedding) {
            l2_norm += val * val;
        }
        l2_norm = std::sqrt(l2_norm);
        if (l2_norm > 1e-6f) {
            for (int i = 0; i < 128; ++i) {
                embedding[i] /= l2_norm;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[AI INFERENCE ERROR] Loi trich xuat embedding: " << e.what() << std::endl;
    }

    return embedding;
}

float FaceEmbedder::calculateSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != 128 || vec2.size() != 128) return 0.0f;
    
    // Do vector đã được L2 Normalized, Cosine Similarity chính là Dot Product
    float dot_product = 0.0f;
    for (size_t i = 0; i < 128; ++i) {
        dot_product += vec1[i] * vec2[i];
    }
    return dot_product;
}
