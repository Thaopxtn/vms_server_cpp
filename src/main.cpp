// main.cpp - Entry point vĂ  HTTP REST API Server cho VMS C++ Core
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <memory>
#include <sstream>
#include <fstream>
#include <optional>
#include <unordered_set>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <zmq.hpp>
#include <pqxx/pqxx>
#include <filesystem>
#include "AnprIntegration.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#undef GET
#undef POST
#else
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#endif

#include "InferenceEngine.hpp"
#include "StorageEngine.hpp"

#ifdef _WIN32
#undef GET
#undef POST
#undef OPTIONS
#endif

// ThÆ° viá»‡n Crow (Header-only micro-framework cho C++)
#include "crow.h"

#ifdef _WIN32
#undef OPTIONS
#endif

// Middleware giáº£i quyáº¿t CORS chĂ©o nguá»“n cho Web Client
#include "CORSMiddleware.hpp"

// Forward declaration cá»§a helper functions
std::string urlEncode(const std::string& value);
std::string encodeRTSPUrlCredentials(const std::string& url);
void drawDetectionOverlay(const std::string& camera_id, cv::Mat& frame, const cv::Size& source_size);

// Biáº¿n Ä‘iÂu khiá»ƒn há»‡ thá»‘ng cháº¡y ngáº§m
std::atomic<bool> g_running{true};
std::atomic<bool> g_ai_enabled{true};
std::atomic<bool> g_ai_face_enabled{true};
std::atomic<bool> g_ai_plate_enabled{true};
std::atomic<bool> g_ai_object_enabled{true};
std::atomic<bool> g_ai_motion_enabled{true};
std::atomic<std::uint64_t> g_ai_batch_generation{0};

std::string g_ai_model_vehicle = "yolo11n.onnx";
std::string g_ai_model_plate = "best_plateSegment.onnx";
std::string g_ai_model_face = "yolo11n-face.onnx";
std::mutex g_ai_model_mutex;

std::mutex g_cameras_mutex;

// Cáº¥u hĂ¬nh káº¿t ná»‘i PostgreSQL (Láº¥y tá»« biáº¿n mĂ´i trÆ°Âng hoáº·c dĂ¹ng máº·c Ä‘á»‹nh)
extern const std::string DB_CONN_STR = []() {
    const char* env_conn = std::getenv("DB_CONN_STR");
    if (env_conn) return std::string(env_conn);
    return std::string("host=localhost port=5432 dbname=smart_monitoring user=postgres password=585858");
}();

// Danh sĂ¡ch camera Ä‘ang hoáº¡t Ä‘á»™ng
struct ActiveCameraInfo {
    std::string camera_id;
    std::string name;
    std::unique_ptr<RTSPCapture> capture;
    std::string stream_id;
};
std::vector<ActiveCameraInfo> g_active_cameras;
// IDs started on demand by the desktop UI; guarded by g_cameras_mutex.
std::unordered_set<std::string> g_runtime_camera_ids;


namespace {
constexpr int kUiPublishFps = 8;
constexpr int kUiMaxFrameWidth = 960;
constexpr int kUiJpegQuality = 70;
constexpr const char* kFramePublisherEndpoint = "tcp://127.0.0.1:5555";

bool envFlagEnabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value || !*value) return default_value;
    std::string normalized{value};
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return normalized == "1" || normalized == "true" || normalized == "yes" ||
           normalized == "on";
}
}

void framePublisherLoop() {
    try {
        zmq::context_t context(1);
        zmq::socket_t publisher(context, zmq::socket_type::pub);
        publisher.set(zmq::sockopt::sndhwm, 2);
        publisher.set(zmq::sockopt::linger, 0);
        publisher.bind(kFramePublisherEndpoint);
        std::cout << "[FRAME SERVICE] ZMQ PUB dang chay tai "
                  << kFramePublisherEndpoint << " (" << kUiPublishFps
                  << " FPS, max width " << kUiMaxFrameWidth << ")" << std::endl;

        const auto interval = std::chrono::milliseconds(1000 / kUiPublishFps);
        auto next_tick = std::chrono::steady_clock::now();

        while (g_running.load()) {
            next_tick += interval;
            std::vector<std::pair<std::string, cv::Mat>> frames;
            {
                std::lock_guard<std::mutex> lock(g_cameras_mutex);
                frames.reserve(g_active_cameras.size());
                for (const auto& camera : g_active_cameras) {
                    if (!camera.capture) continue;
                    auto frame = camera.capture->getLatestFrame();
                    if (!frame.empty()) {
                        frames.emplace_back(camera.camera_id, std::move(frame));
                    }
                }
            }

            for (auto& [camera_id, frame] : frames) {
                cv::Mat ui_frame;
                if (frame.cols > kUiMaxFrameWidth) {
                    const auto scale = static_cast<double>(kUiMaxFrameWidth) /
                                       static_cast<double>(frame.cols);
                    cv::resize(frame, ui_frame,
                               cv::Size(kUiMaxFrameWidth,
                                        static_cast<int>(frame.rows * scale)),
                               0.0, 0.0, cv::INTER_AREA);
                } else {
                    ui_frame = frame;
                }

                drawDetectionOverlay(camera_id, ui_frame, frame.size());

                std::vector<unsigned char> jpeg;
                const std::vector<int> params{
                    cv::IMWRITE_JPEG_QUALITY, kUiJpegQuality
                };
                if (!cv::imencode(".jpg", ui_frame, jpeg, params)) continue;

                zmq::message_t topic(camera_id.data(), camera_id.size());
                zmq::message_t payload(jpeg.data(), jpeg.size());
                const auto topic_sent = publisher.send(topic, zmq::send_flags::sndmore);
                if (topic_sent) {
                    const auto payload_sent = publisher.send(payload, zmq::send_flags::none);
                    (void)payload_sent;
                }
            }

            std::this_thread::sleep_until(next_tick);
        }

        publisher.close();
        context.close();
    } catch (const std::exception& error) {
        std::cerr << "[FRAME SERVICE ERROR] " << error.what() << std::endl;
    }
}

// Quáº£n lĂ½ lÆ°u trá»¯ NVR
std::unique_ptr<NVRStorageEngine> g_nvr_engine;

struct RetiredCapture {
    std::string camera_id;
    std::unique_ptr<RTSPCapture> capture;
    bool finalize_recording{false};
};

std::mutex g_retired_captures_mutex;
std::condition_variable g_retired_captures_cv;
std::deque<RetiredCapture> g_retired_captures;
std::atomic<bool> g_capture_cleanup_shutdown{false};

void retireCapture(
    const std::string& camera_id,
    std::unique_ptr<RTSPCapture> capture,
    bool finalize_recording) {
    if (!capture) {
        if (finalize_recording && g_nvr_engine) {
            g_nvr_engine->closeCamera(camera_id);
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_retired_captures_mutex);
        g_retired_captures.push_back(
            {camera_id, std::move(capture), finalize_recording});
    }
    g_retired_captures_cv.notify_one();
}

void captureCleanupLoop() {
    while (true) {
        RetiredCapture retired;
        {
            std::unique_lock<std::mutex> lock(g_retired_captures_mutex);
            g_retired_captures_cv.wait(lock, [] {
                return g_capture_cleanup_shutdown.load() ||
                       !g_retired_captures.empty();
            });
            if (g_capture_cleanup_shutdown.load() &&
                g_retired_captures.empty()) {
                return;
            }
            retired = std::move(g_retired_captures.front());
            g_retired_captures.pop_front();
        }
        if (retired.capture) retired.capture->stop();
        if (retired.finalize_recording && g_nvr_engine) {
            g_nvr_engine->closeCamera(retired.camera_id);
        }
    }
}

// AI Detector toĂ n cá»¥c
std::unique_ptr<ONNXDetector> g_face_detector;
std::unique_ptr<ONNXDetector> g_plate_detector;
std::unique_ptr<ONNXDetector> g_vehicle_detector; // YOLOv11 general (car, motorbike, person)
std::unique_ptr<FaceEmbedder> g_face_embedder;

// Cache bounding box AI detection cho má»—i camera (phá»¥c vá»¥ overlay realtime trĂªn web)
struct DetectionBox {
    int x, y, w, h;
    std::string label;
    float confidence;
};
struct CameraDetections {
    std::string camera_id;
    std::vector<DetectionBox> boxes;
    long long timestamp;
    int frame_width;
    int frame_height;
};
std::map<std::string, CameraDetections> g_detections_cache;
std::mutex g_detections_mutex;

void drawDetectionOverlay(const std::string& camera_id, cv::Mat& frame, const cv::Size& source_size) {
    CameraDetections detections;
    {
        std::lock_guard<std::mutex> lock(g_detections_mutex);
        const auto found = g_detections_cache.find(camera_id);
        if (found == g_detections_cache.end() || time(nullptr) - found->second.timestamp > 3) return;
        detections = found->second;
    }
    if (source_size.width <= 0 || source_size.height <= 0) return;
    const auto scale_x = static_cast<double>(frame.cols) / source_size.width;
    const auto scale_y = static_cast<double>(frame.rows) / source_size.height;
    for (const auto& box : detections.boxes) {
        const cv::Rect scaled{
            static_cast<int>(box.x * scale_x), static_cast<int>(box.y * scale_y),
            static_cast<int>(box.w * scale_x), static_cast<int>(box.h * scale_y)};
        const auto clipped = scaled & cv::Rect{0, 0, frame.cols, frame.rows};
        if (clipped.empty()) continue;
        cv::rectangle(frame, clipped, cv::Scalar{0, 220, 80}, 2);
        const auto label = box.label + " " + std::to_string(static_cast<int>(box.confidence * 100.0f)) + "%";
        cv::putText(frame, label, cv::Point{clipped.x, std::max(18, clipped.y - 6)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar{0, 220, 80}, 1, cv::LINE_AA);
    }
}

struct AiRuntimeSummary {
    int active_cameras{0};
    int live_boxes{0};
    int last_frame_age_seconds{-1};
};

AiRuntimeSummary getAiRuntimeSummary() {
    AiRuntimeSummary summary;
    const auto now = time(nullptr);
    time_t latest_frame{0};
    std::lock_guard<std::mutex> lock(g_detections_mutex);
    for (const auto& [camera_id, detection] : g_detections_cache) {
        if (now - detection.timestamp > 5) continue;
        ++summary.active_cameras;
        summary.live_boxes += static_cast<int>(detection.boxes.size());
        latest_frame = std::max(latest_frame, static_cast<time_t>(detection.timestamp));
    }
    if (latest_frame > 0) {
        summary.last_frame_age_seconds = static_cast<int>(now - latest_frame);
    }
    return summary;
}

// Bá»™ nhá»› Ä‘á»‡m cho nháº­n diá»‡n khuĂ´n máº·t (RAM Cache)
struct FaceIdentity {
    std::string label;
    bool is_whitelisted;
    std::vector<float> embedding;
};
std::vector<FaceIdentity> g_face_identities_cache;
std::mutex g_identities_mutex;

// Danh sĂ¡ch biá»ƒn sá»‘ Ä‘Ă£ xá»­ lĂ½ gáº§n Ä‘Ă¢y (dedup - trĂ¡nh lÆ°u trĂ¹ng láº·p)
std::unordered_map<std::string, time_t> g_plate_dedup_cache; // plate_text -> last_seen_time
std::mutex g_dedup_mutex;
const int PLATE_DEDUP_SECONDS = 30; // Biá»ƒn sá»‘ láº·p láº¡i trong 30s thĂ¬ bÂ qua

// Danh sĂ¡ch camera Ä‘ang báº­t phĂ¢n tĂ­ch AI (on-demand)
std::unordered_set<std::string> g_ai_camera_list; // camera_id
std::mutex g_ai_list_mutex;

// TĂªn vá»‹ trĂ­ cá»§a cĂ¡c camera (cache tá»« DB)
std::unordered_map<std::string, std::string> g_camera_location_cache; // camera_id -> location_name
std::mutex g_location_mutex;

// Quáº£n lĂ½ káº¿t ná»‘i WebSockets thÂi gian thá»±c
std::unordered_set<crow::websocket::connection*> g_ws_connections;
std::mutex g_ws_mutex;

// ThĂ´ng tin tiáº¿n trĂ¬nh go2rtc sidecar
#ifdef _WIN32
PROCESS_INFORMATION g_go2rtc_pi;
#else
pid_t g_go2rtc_pid = -1;
#endif

// ThĂ´ng tin tiáº¿n trĂ¬nh status poller cháº¡y nÂn
#ifdef _WIN32
PROCESS_INFORMATION g_poller_pi;
HANDLE g_poller_job = nullptr;
#else
pid_t g_poller_pid = -1;
#endif

// ThĂ´ng tin tiáº¿n trĂ¬nh OCR service
#ifdef _WIN32
PROCESS_INFORMATION g_ocr_pi;
#else
pid_t g_ocr_pid = -1;
#endif

// Thá»‘ng kĂª AI realtime
struct AIStats {
    std::atomic<int> total_vehicles{0};
    std::atomic<int> total_plates{0};
    std::atomic<int> total_faces{0};
    std::atomic<int> total_persons{0};
    std::atomic<int> frames_processed{0};
    std::atomic<std::uint64_t> plate_attempts{0};
    std::atomic<std::uint64_t> plate_boxes_detected{0};
    std::atomic<std::uint64_t> plate_retry_successes{0};
    std::atomic<std::uint64_t> motion_frames_checked{0};
    std::atomic<std::uint64_t> motion_frames_triggered{0};
    std::atomic<std::uint64_t> motion_frames_skipped{0};
    std::atomic<std::uint64_t> motion_frames_queued{0};
};
AIStats g_ai_stats;

// â”€â”€â”€ PHĂ‚N PHÂI TIáº¾N TRĂŒNH GO2RTC SIDECAR â”€â”€â”€

// Táº¡o tá»‡p cáº¥u hĂ¬nh go2rtc.yaml tá»« cÆ¡ sá»Ÿ dá»¯ liá»‡u
std::string findFFmpegExecutable() {
    if (const char* configured = std::getenv("VMS_FFMPEG_BIN")) {
        if (*configured && std::filesystem::is_regular_file(configured)) return configured;
    }
    if (std::filesystem::is_regular_file("ffmpeg.exe")) return "ffmpeg.exe";
#ifdef _WIN32
    if (const char* local_app_data = std::getenv("LOCALAPPDATA")) {
        const auto packages = std::filesystem::path{local_app_data} / "Microsoft" / "WinGet" / "Packages";
        std::error_code error;
        if (std::filesystem::is_directory(packages, error)) {
            for (const auto& package : std::filesystem::directory_iterator(packages, error)) {
                if (error) break;
                if (!package.is_directory() || package.path().filename().string().rfind("Gyan.FFmpeg_", 0) != 0) continue;
                for (const auto& file : std::filesystem::recursive_directory_iterator(
                         package.path(), std::filesystem::directory_options::skip_permission_denied, error)) {
                    if (error) break;
                    if (file.is_regular_file() && file.path().filename() == "ffmpeg.exe") {
                        return file.path().generic_string();
                    }
                }
            }
        }
    }
#endif
    return "ffmpeg";
}

void generateGo2RTCConfig(const std::vector<std::pair<std::string, std::string>>& cameras) {
    auto writeYaml = [&](const std::string& path) {
        std::ofstream yaml_file(path);
        if (!yaml_file.is_open()) return;
        yaml_file << "streams:\n";
        for (const auto& [cam_id, rtsp_url] : cameras) {
            yaml_file << "  " << cam_id << ": " << rtsp_url << "\n";
            
            std::string hd_url = rtsp_url;
            size_t pos = hd_url.find("subtype=1");
            if (pos != std::string::npos) {
                hd_url.replace(pos, 9, "subtype=0");
            } else if (hd_url.find("subtype=0") == std::string::npos) {
                if (hd_url.find("?") != std::string::npos) {
                    hd_url += "&subtype=0";
                } else {
                    hd_url += "?subtype=0";
                }
            }
            yaml_file << "  " << cam_id << "_hd: " << hd_url << "\n";
        }
        yaml_file << "\napi:\n  listen: \":1984\"\n  origin: \"*\"\n";
        yaml_file << "\nwebrtc:\n  listen: \":8555\"\n";
        yaml_file << "\nffmpeg:\n  bin: \"" << findFFmpegExecutable() << "\"\n";
        yaml_file.close();
    };

    writeYaml("go2rtc.yaml");
    writeYaml("build/Release/go2rtc.yaml");
    writeYaml("../go2rtc.yaml");
    writeYaml("../../go2rtc.yaml");
    std::cout << "[VMS SIDE] Da ghi file go2rtc.yaml voi " << cameras.size() << " luong camera (all paths)." << std::endl;
}

void registerStreamToGo2RTC(const std::string& cam_id, const std::string& rtsp_url) {
    std::string safe_rtsp = encodeRTSPUrlCredentials(rtsp_url);
#ifdef _WIN32
    std::string curl_cmd1 = "curl.exe -s --connect-timeout 2 --max-time 5 -X PUT -G \"http://localhost:1984/api/streams\" --data-urlencode \"name=" + cam_id + "\" --data-urlencode \"src=" + safe_rtsp + "\"";
#else
    std::string curl_cmd1 = "curl -s --connect-timeout 2 --max-time 5 -X PUT -G \"http://localhost:1984/api/streams\" --data-urlencode \"name=" + cam_id + "\" --data-urlencode \"src=" + safe_rtsp + "\"";
#endif
    system(curl_cmd1.c_str());

    std::string hd_url = rtsp_url;
    size_t pos = hd_url.find("subtype=1");
    if (pos != std::string::npos) {
        hd_url.replace(pos, 9, "subtype=0");
    } else if (hd_url.find("subtype=0") == std::string::npos) {
        if (hd_url.find("?") != std::string::npos) hd_url += "&subtype=0";
        else hd_url += "?subtype=0";
    }
    std::string safe_hd = encodeRTSPUrlCredentials(hd_url);
#ifdef _WIN32
    std::string curl_cmd2 = "curl.exe -s --connect-timeout 2 --max-time 5 -X PUT -G \"http://localhost:1984/api/streams\" --data-urlencode \"name=" + cam_id + "_hd\" --data-urlencode \"src=" + safe_hd + "\"";
#else
    std::string curl_cmd2 = "curl -s --connect-timeout 2 --max-time 5 -X PUT -G \"http://localhost:1984/api/streams\" --data-urlencode \"name=" + cam_id + "_hd\" --data-urlencode \"src=" + safe_hd + "\"";
#endif
    system(curl_cmd2.c_str());
}

// Khá»Ÿi cháº¡y tiáº¿n trĂ¬nh go2rtc
void startGo2RTCSidecar() {
#ifdef _WIN32
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE; // áº¨n cá»­a sá»• Console cá»§a go2rtc Ä‘á»ƒ giá»¯ mĂ n hĂ¬nh gÂn gĂ ng
    ZeroMemory(&g_go2rtc_pi, sizeof(g_go2rtc_pi));
    std::string cmd = "go2rtc.exe";
    if (!CreateProcessA(NULL, const_cast<char*>(cmd.c_str()), NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &si, &g_go2rtc_pi)) {
        std::cerr << "[VMS SIDE WARNING] Khong the tu dong khoi chay go2rtc.exe. Vui long dam bao tep nay ton tai trong thu muc build/Release." << std::endl;
    } else {
        std::cout << "[VMS SIDE] Da khoi chay go2rtc sidecar engine (PID: " << g_go2rtc_pi.dwProcessId << ")." << std::endl;
    }
#else
    g_go2rtc_pid = fork();
    if (g_go2rtc_pid == 0) {
        execlp("./go2rtc", "./go2rtc", NULL);
        exit(1);
    } else if (g_go2rtc_pid > 0) {
        std::cout << "[VMS SIDE] Da khoi chay go2rtc sidecar engine (PID: " << g_go2rtc_pid << ")." << std::endl;
    }
#endif
}

// Dá»«ng tiáº¿n trĂ¬nh go2rtc khi táº¯t server
void stopGo2RTCSidecar() {
#ifdef _WIN32
    if (g_go2rtc_pi.hProcess) {
        TerminateProcess(g_go2rtc_pi.hProcess, 0);
        CloseHandle(g_go2rtc_pi.hProcess);
        CloseHandle(g_go2rtc_pi.hThread);
        std::cout << "[VMS SIDE] Da dung go2rtc sidecar engine." << std::endl;
    }
#else
    if (g_go2rtc_pid > 0) {
        kill(g_go2rtc_pid, SIGTERM);
        std::cout << "[VMS SIDE] Da dung go2rtc sidecar engine." << std::endl;
    }
#endif
}

// TĂ¬m Ä‘Æ°Âng dáº«n thá»±c táº¿ cá»§a file ká»‹ch báº£n python dá»±a trĂªn vá»‹ trĂ­ thÆ° má»¥c cháº¡y hiá»‡n táº¡i
std::string getPythonScriptPath(const std::string& script_name) {
    if (std::ifstream(script_name).good()) {
        return script_name;
    }
    if (std::ifstream("../" + script_name).good()) {
        return "../" + script_name;
    }
    if (std::ifstream("../../" + script_name).good()) {
        return "../../" + script_name;
    }
    return script_name;
}

// Khá»Ÿi cháº¡y tiáº¿n trĂ¬nh status poller
void startStatusPoller() {
#ifdef _WIN32
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESHOWWINDOW;
    startup.wShowWindow = SW_HIDE;
    ZeroMemory(&g_poller_pi, sizeof(g_poller_pi));
    std::string command = "python " + getPythonScriptPath("nvr_status_poller.py");
    g_poller_job = CreateJobObjectA(nullptr, nullptr);
    if (g_poller_job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(g_poller_job, JobObjectExtendedLimitInformation,
                                &limits, sizeof(limits));
    }
    if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr,
                        &startup, &g_poller_pi)) {
        std::cerr << "[VMS SIDE WARNING] Khong the khoi dong nvr_status_poller.py.\n";
        if (g_poller_job) CloseHandle(g_poller_job);
        g_poller_job = nullptr;
        return;
    }
    if (g_poller_job &&
        !AssignProcessToJobObject(g_poller_job, g_poller_pi.hProcess)) {
        TerminateProcess(g_poller_pi.hProcess, 0);
        CloseHandle(g_poller_pi.hProcess);
        CloseHandle(g_poller_pi.hThread);
        ZeroMemory(&g_poller_pi, sizeof(g_poller_pi));
        CloseHandle(g_poller_job);
        g_poller_job = nullptr;
        return;
    }
    ResumeThread(g_poller_pi.hThread);
    std::cout << "[VMS SIDE] nvr_status_poller.py started (PID "
              << g_poller_pi.dwProcessId << ").\n";
#else
    g_poller_pid = fork();
    if (g_poller_pid == 0) {
        execlp("python", "python", "nvr_status_poller.py", NULL);
        exit(1);
    } else if (g_poller_pid > 0) {
        std::cout << "[VMS SIDE] Da khoi chay nvr_status_poller.py (PID: " << g_poller_pid << ")." << std::endl;
    }
#endif
}

void stopStatusPoller() {
#ifdef _WIN32
    if (g_poller_pi.hProcess) {
        TerminateProcess(g_poller_pi.hProcess, 0);
        WaitForSingleObject(g_poller_pi.hProcess, 1000);
        CloseHandle(g_poller_pi.hProcess);
        CloseHandle(g_poller_pi.hThread);
        ZeroMemory(&g_poller_pi, sizeof(g_poller_pi));
    }
    if (g_poller_job) {
        CloseHandle(g_poller_job);
        g_poller_job = nullptr;
    }
#else
    if (g_poller_pid > 0) {
        kill(g_poller_pid, SIGTERM);
        std::cout << "[VMS SIDE] Da dung nvr_status_poller.py." << std::endl;
    }
#endif
}

// Khá»Ÿi cháº¡y tiáº¿n trĂ¬nh OCR Service
void startOCRService() {
#ifdef _WIN32
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    ZeroMemory(&g_ocr_pi, sizeof(g_ocr_pi));
    std::string cmd = "python supervision_ai_service.py";
    if (!CreateProcessA(NULL, const_cast<char*>(cmd.c_str()), NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &g_ocr_pi)) {
        std::cerr << "[VMS SIDE WARNING] Khong the khoi chay supervision_ai_service.py. Bien so se khong duoc doc ky tu." << std::endl;
    } else {
        std::cout << "[VMS SIDE] Da khoi chay supervision_ai_service.py (PID: " << g_ocr_pi.dwProcessId << ")." << std::endl;
    }
#else
    g_ocr_pid = fork();
    if (g_ocr_pid == 0) {
        execlp("python", "python", "supervision_ai_service.py", NULL);
        exit(1);
    } else if (g_ocr_pid > 0) {
        std::cout << "[VMS SIDE] Da khoi chay supervision_ai_service.py (PID: " << g_ocr_pid << ")." << std::endl;
    }
#endif
}

// Dá»«ng tiáº¿n trĂ¬nh OCR Service
void stopOCRService() {
#ifdef _WIN32
    if (g_ocr_pi.hProcess) {
        TerminateProcess(g_ocr_pi.hProcess, 0);
        CloseHandle(g_ocr_pi.hProcess);
        CloseHandle(g_ocr_pi.hThread);
        std::cout << "[VMS SIDE] Da dung supervision_ai_service.py." << std::endl;
    }
#else
    if (g_ocr_pid > 0) {
        kill(g_ocr_pid, SIGTERM);
        std::cout << "[VMS SIDE] Da dung supervision_ai_service.py." << std::endl;
    }
#endif
}

// â”€â”€â”€ HELPER FUNCTIONS â”€â”€â”€

// HĂ m URL Encode Ä‘á»ƒ truyÂn tham sá»‘ an toĂ n trong curl CLI
std::string urlEncode(const std::string& value) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (std::string::const_iterator i = value.begin(), n = value.end(); i != n; ++i) {
        std::string::value_type c = (*i);

        // Giá»¯ nguyĂªn cĂ¡c kĂ½ tá»± chá»¯ vĂ  sá»‘ thĂ´ng thÆ°Âng
        if (isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
            continue;
        }

        // MĂ£ hĂ³a táº¥t cáº£ cĂ¡c kĂ½ tá»± khĂ¡c
        escaped << '%' << std::setw(2) << int((unsigned char)c);
    }

    return escaped.str();
}

// HĂ m chuáº©n hĂ³a ID Ä‘á»ƒ trĂ¡nh cĂ¡c kĂ½ tá»± Ä‘áº·c biá»‡t

std::string encodeRTSPUrlCredentials(const std::string& url) {
    size_t scheme_pos = url.find("://");
    if (scheme_pos == std::string::npos) return url;
    std::string prefix = url.substr(0, scheme_pos + 3);
    
    size_t path_pos = url.find('/', prefix.length());
    std::string authority = url.substr(prefix.length(), path_pos != std::string::npos ? path_pos - prefix.length() : std::string::npos);
    
    size_t at_pos = authority.rfind('@');
    if (at_pos == std::string::npos) return url;
    
    std::string creds = authority.substr(0, at_pos);
    size_t colon_pos = creds.find(':');
    
    std::string user = creds, pass = "";
    if (colon_pos != std::string::npos) {
        user = creds.substr(0, colon_pos);
        pass = creds.substr(colon_pos + 1);
    }
    
    // Quick decode to prevent double encoding
    auto urlDecode = [](const std::string& str) {
        std::string ret;
        char ch;
        int i, ii, len = str.length();
        for (i=0; i < len; i++){
            if(str[i] != '%'){
                if(str[i] == '+') ret += ' ';
                else ret += str[i];
            }else{
                if (i + 2 < len) {
                    sscanf(str.substr(i + 1, 2).c_str(), "%x", &ii);
                    ch = static_cast<char>(ii);
                    ret += ch;
                    i = i + 2;
                } else {
                    ret += str[i];
                }
            }
        }
        return ret;
    };
    
    std::string encoded_user = urlEncode(urlDecode(user));
    std::string encoded_pass = pass.empty() ? "" : ":" + urlEncode(urlDecode(pass));
    
    std::string new_url = prefix + encoded_user + encoded_pass + "@" + authority.substr(at_pos + 1);
    if (path_pos != std::string::npos) {
        new_url += url.substr(path_pos);
    }
    return new_url;
}

std::string sanitizeId(const std::string& input) {
    std::string result = "";
    for (char c : input) {
        if (isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
            result += c;
        } else if (c == ' ') {
            result += '_';
        }
    }
    if (result.empty()) result = "NVR_CAM";
    return result;
}

// Thá»±c hiá»‡n quĂ©t vĂ  Ä‘á»“ng bá»™ cĂ¡c kĂªnh camera tá»« Ä‘áº§u ghi NVR Dahua/Hikvision

// Helper function to prevent using SDK/TCP ports for RTSP
int normalizeRtspPort(int port) {
    if (port >= 37700 && port <= 37899) return 554; // Dahua TCP ports
    if (port == 8000) return 554; // Hikvision SDK port
    if (port == 34567) return 554; // XMeye TCP port
    return port;
}

int syncDVRCameras(const std::string& dvr_id, const std::string& host, int http_port, int rtsp_port, const std::string& username, const std::string& password, const std::string& brand) {
    rtsp_port = normalizeRtspPort(rtsp_port);
    std::string clean_host = host;
    // BÂ http:// hoáº·c https:// hoáº·c dáº¥u gáº¡ch chĂ©o á»Ÿ cuá»‘i náº¿u cĂ³
    if (clean_host.rfind("http://", 0) == 0) clean_host = clean_host.substr(7);
    else if (clean_host.rfind("https://", 0) == 0) clean_host = clean_host.substr(8);
    if (!clean_host.empty() && clean_host.back() == '/') clean_host.pop_back();

    std::string cmd;
    if (brand == "dahua" || brand == "generic") {
        // DĂ¹ng curl.exe gÂi Dahua CGI API Ä‘á»ƒ láº¥y danh sĂ¡ch kĂªnh
        cmd = "curl.exe --digest -s -u " + username + ":" + password + " \"http://" + clean_host + ":" + std::to_string(http_port) + "/cgi-bin/configManager.cgi?action=getConfig&name=ChannelTitle\"";
    } else {
        // HĂ£ng khĂ¡c (vĂ­ dá»¥ Hikvision cĂ³ thá»ƒ má»Ÿ rá»™ng API ISAPI á»Ÿ Ä‘Ă¢y)
        std::cout << "[VMS SYNC] Quet mac dinh 16 kenh cho dau ghi Hikvision..." << std::endl;
        int added = 0;
        try {
            pqxx::connection conn(DB_CONN_STR);
            std::string clean_dvr_id = sanitizeId(dvr_id);
            for (int i = 1; i <= 16; ++i) {
                std::string cam_id = clean_dvr_id + "_CAM_" + (i < 10 ? "0" : "") + std::to_string(i);
                std::string cam_name = "Camera " + std::to_string(i);
                std::string encoded_user = urlEncode(username);
                std::string encoded_pass = urlEncode(password);
                std::string rtsp_url = "rtsp://" + encoded_user + ":" + encoded_pass + "@" + clean_host + ":" + std::to_string(rtsp_port) + "/Streaming/Channels/" + std::to_string(i) + "02";
                pqxx::work txn(conn);
                conn.prepare("check_cam_" + std::to_string(i), "SELECT 1 FROM cameras WHERE camera_id = $1");
                pqxx::result r = txn.exec_prepared("check_cam_" + std::to_string(i), cam_id);
                if (r.empty()) {
                    conn.prepare("ins_cam_" + std::to_string(i), "INSERT INTO cameras (camera_id, dvr_id, name, rtsp_url) VALUES ($1, $2, $3, $4)");
                    txn.exec_prepared("ins_cam_" + std::to_string(i), cam_id, dvr_id, cam_name, rtsp_url);
                    added++;
                }
                txn.commit();
            }
        } catch (...) {}
        return added;
    }

    std::cout << "[VMS SYNC] Dang chay lenh quet: " << cmd << std::endl;
    // Thá»±c thi curl vĂ  Ä‘Âc káº¿t quáº£
    std::string output = "";
    char buffer[512];
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) {
        std::cerr << "[VMS SYNC ERROR] Khong the chay curl.exe" << std::endl;
        return 0;
    }
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    int added_count = 0;
    std::stringstream ss(output);
    std::string line;
    std::string clean_dvr_id = sanitizeId(dvr_id);
    try {
        pqxx::connection conn(DB_CONN_STR);
        while (std::getline(ss, line)) {
            // Âá»‹nh dáº¡ng: table.ChannelTitle[i].Name=Name
            if (line.rfind("table.ChannelTitle[", 0) == 0) {
                size_t start_idx = 19;
                size_t end_idx = line.find("].Name=", start_idx);
                if (end_idx != std::string::npos) {
                    std::string idx_str = line.substr(start_idx, end_idx - start_idx);
                    int channel_idx = std::stoi(idx_str);
                    std::string cam_name = line.substr(end_idx + 7);
                    if (!cam_name.empty() && cam_name.back() == '\r') cam_name.pop_back();
                    if (!cam_name.empty() && cam_name.back() == '\n') cam_name.pop_back();
                    if (cam_name.empty()) continue;

                    std::string cam_id = clean_dvr_id + "_CAM_" + (channel_idx + 1 < 10 ? "0" : "") + std::to_string(channel_idx + 1);
                    std::string encoded_user = urlEncode(username);
                    std::string encoded_pass = urlEncode(password);
                    std::string rtsp_url = "rtsp://" + encoded_user + ":" + encoded_pass + "@" + clean_host + ":" + std::to_string(rtsp_port) + "/cam/realmonitor?channel=" + std::to_string(channel_idx + 1) + "&subtype=1";
                    
                    pqxx::work txn(conn);
                    conn.prepare("check_cam_" + idx_str, "SELECT 1 FROM cameras WHERE camera_id = $1");
                    pqxx::result r = txn.exec_prepared("check_cam_" + idx_str, cam_id);
                    if (r.empty()) {
                        conn.prepare("ins_cam_" + idx_str, "INSERT INTO cameras (camera_id, dvr_id, name, rtsp_url) VALUES ($1, $2, $3, $4)");
                        txn.exec_prepared("ins_cam_" + idx_str, cam_id, dvr_id, cam_name, rtsp_url);
                        added_count++;
                    }
                    txn.commit();

                    // LUĂ”N Ä‘Äƒng kĂ½ luá»“ng vá»›i go2rtc á»Ÿ runtime Ä‘á»ƒ Ä‘áº£m báº£o luá»“ng hoáº¡t Ä‘á»™ng ngay láº­p tá»©c
                    registerStreamToGo2RTC(cam_id, rtsp_url);
                    std::cout << "[VMS SYNC] Dang ky luong (SD va HD) " << cam_id << " len go2rtc" << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[VMS SYNC ERROR] Loi ghi camera vao DB: " << e.what() << std::endl;
    }
    std::cout << "[VMS SYNC] Da dong bo thanh cong " << added_count << " camera tu dau ghi " << dvr_id << std::endl;
    return added_count;
}

// Náº¡p danh sĂ¡ch vector khuĂ´n máº·t Ä‘Äƒng kĂ½ vĂ o RAM cache
void loadFaceIdentitiesCache() {
    std::lock_guard<std::mutex> lock(g_identities_mutex);
    g_face_identities_cache.clear();
    try {
        pqxx::connection conn(DB_CONN_STR);
        pqxx::nontransaction txn(conn);
        pqxx::result r = txn.exec("SELECT label, is_whitelisted, face_embedding FROM identities WHERE identity_type = 'face' AND face_embedding IS NOT NULL");
        for (auto row : r) {
            std::string label = row["label"].as<std::string>();
            bool is_whitelisted = row["is_whitelisted"].as<bool>();
            std::string emb_str = row["face_embedding"].as<std::string>();
            std::vector<float> embedding;
            std::stringstream ss(emb_str);
            std::string val;
            while (std::getline(ss, val, ',')) {
                try {
                    embedding.push_back(std::stof(val));
                } catch (...) {}
            }
            if (embedding.size() == 128) {
                g_face_identities_cache.push_back({label, is_whitelisted, embedding});
            }
        }
        std::cout << "[DATABASE] Da nap " << g_face_identities_cache.size() << " khuon mat mau vao bo nho dem (RAM Cache)." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[DATABASE ERROR] Khong the nap danh sach khuon mat: " << e.what() << std::endl;
    }
}

// Khá»Ÿi táº¡o cĂ¡c báº£ng PostgreSQL
bool initDatabase() {
    try {
        pqxx::connection conn(DB_CONN_STR);
        pqxx::work txn(conn);

        // Táº¡o báº£ng dvrs trÆ°á»›c Ä‘á»ƒ cĂ³ khĂ³a ngoáº¡i
        txn.exec(
            "CREATE TABLE IF NOT EXISTS dvrs ("
            "    id SERIAL PRIMARY KEY,"
            "    dvr_id VARCHAR(50) UNIQUE NOT NULL,"
            "    name VARCHAR(100) NOT NULL,"
            "    host VARCHAR(100) NOT NULL,"
            "    http_port INTEGER DEFAULT 80,"
            "    rtsp_port INTEGER DEFAULT 554,"
            "    tcp_port INTEGER DEFAULT 37777,"
            "    username VARCHAR(50) DEFAULT 'admin',"
            "    password VARCHAR(100) NOT NULL,"
            "    brand VARCHAR(50) DEFAULT 'generic',"
            "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ");"
        );
        txn.exec("ALTER TABLE dvrs ADD COLUMN IF NOT EXISTS tcp_port INTEGER DEFAULT 37777;");


        // Táº¡o báº£ng cameras (cĂ³ liĂªn káº¿t dvr_id)
        txn.exec(
            "CREATE TABLE IF NOT EXISTS cameras ("
            "    id SERIAL PRIMARY KEY,"
            "    camera_id VARCHAR(50) UNIQUE NOT NULL,"
            "    dvr_id VARCHAR(50) REFERENCES dvrs(dvr_id) ON DELETE SET NULL,"
            "    name VARCHAR(100) NOT NULL,"
            "    rtsp_url VARCHAR(500) NOT NULL,"
            "    is_active BOOLEAN DEFAULT TRUE,"
            "    grid_position INTEGER DEFAULT -1,"
            "    latitude DOUBLE PRECISION,"
            "    longitude DOUBLE PRECISION,"
            "    is_public BOOLEAN DEFAULT FALSE,"
            "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ");"
        );

        // Âáº£m báº£o cá»™t dvr_id tá»“n táº¡i
        txn.exec("ALTER TABLE cameras ADD COLUMN IF NOT EXISTS dvr_id VARCHAR(50) REFERENCES dvrs(dvr_id) ON DELETE SET NULL;");

        // Táº¡o báº£ng events
        txn.exec(
            "CREATE TABLE IF NOT EXISTS events ("
            "    id SERIAL PRIMARY KEY,"
            "    camera_id VARCHAR(50) NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,"
            "    event_type VARCHAR(50) NOT NULL,"
            "    description TEXT DEFAULT '',"
            "    confidence DOUBLE PRECISION DEFAULT 0.0,"
            "    snapshot_path VARCHAR(500) DEFAULT '',"
            "    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ");"
        );

        txn.exec("ALTER TABLE events ADD COLUMN IF NOT EXISTS plate_text VARCHAR(20) DEFAULT '';");
        txn.exec("ALTER TABLE events ADD COLUMN IF NOT EXISTS plate_image_path VARCHAR(500) DEFAULT '';");

        // Táº¡o báº£ng identities
        txn.exec(
            "CREATE TABLE IF NOT EXISTS identities ("
            "    id SERIAL PRIMARY KEY,"
            "    identity_type VARCHAR(20) NOT NULL,"
            "    label VARCHAR(100) NOT NULL,"
            "    data_path VARCHAR(500) DEFAULT '',"
            "    is_whitelisted BOOLEAN DEFAULT FALSE,"
            "    face_embedding TEXT,"
            "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ");"
        );

        // Táº¡o báº£ng detections - lÆ°u káº¿t quáº£ AI chi tiáº¿t (biá»ƒn sá»‘, phÆ°Æ¡ng tiá»‡n, ngÆ°Âi)
        txn.exec(
            "CREATE TABLE IF NOT EXISTS detections ("
            "    id SERIAL PRIMARY KEY,"
            "    camera_id VARCHAR(50) NOT NULL,"
            "    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "    object_type VARCHAR(30) NOT NULL,"
            "    plate_text VARCHAR(20) DEFAULT '',"
            "    plate_confidence FLOAT DEFAULT 0.0,"
            "    identity_label VARCHAR(100) DEFAULT '',"
            "    identity_confidence FLOAT DEFAULT 0.0,"
            "    vehicle_image_path VARCHAR(500) DEFAULT '',"
            "    plate_image_path VARCHAR(500) DEFAULT '',"
            "    face_image_path VARCHAR(500) DEFAULT '',"
            "    bbox_x INTEGER DEFAULT 0,"
            "    bbox_y INTEGER DEFAULT 0,"
            "    bbox_w INTEGER DEFAULT 0,"
            "    bbox_h INTEGER DEFAULT 0,"
            "    location_name VARCHAR(200) DEFAULT '',"
            "    extra_data TEXT DEFAULT ''"
            ");"
        );

        // Index Ä‘á»ƒ tĂ¬m kiáº¿m nhanh theo camera vĂ  thÂi gian
        txn.exec("CREATE INDEX IF NOT EXISTS idx_detections_camera ON detections(camera_id);");
        txn.exec("CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(detected_at DESC);");
        txn.exec("CREATE INDEX IF NOT EXISTS idx_detections_plate ON detections(plate_text);");
        txn.exec("CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(object_type);");

        txn.commit();
        std::cout << "[DATABASE] Da khoi tao CSDL PostgreSQL thanh cong." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[DATABASE ERROR] Khong the khoi tao CSDL: " << e.what() << std::endl;
        return false;
    }
}

// LÆ°u log sá»± kiá»‡n AI phĂ¡t hiá»‡n vĂ  phĂ¡t Ä‘i cáº£nh bĂ¡o WebSocket tá»©c thÂi
void saveEvent(const std::string& camera_id, const std::string& type, const std::string& desc, float conf, const std::string& snap_path, const std::string& plate_text = "", const std::string& plate_image_path = "") {
    try {
        pqxx::connection conn(DB_CONN_STR);
        pqxx::work txn(conn);
        conn.prepare("insert_event", "INSERT INTO events (camera_id, event_type, description, confidence, snapshot_path, plate_text, plate_image_path) VALUES ($1, $2, $3, $4, $5, $6, $7)");
        txn.exec_prepared("insert_event", camera_id, type, desc, conf, snap_path, plate_text, plate_image_path);
        txn.commit();
        std::cout << "[EVENT SAVED] Camera: " << camera_id << " | Type: " << type << " | Confidence: " << conf << std::endl;

        // ÂĂ³ng gĂ³i dá»¯ liá»‡u JSON gá»­i qua WebSocket
        crow::json::wvalue alert;
        alert["event_type"] = type;
        alert["camera_id"] = camera_id;
        alert["description"] = desc;
        alert["confidence"] = conf;
        alert["snapshot_path"] = snap_path;
        alert["plate_text"] = plate_text;
        alert["plate_image_path"] = plate_image_path;
        alert["timestamp"] = time(nullptr);
        std::string alert_str = alert.dump();

        // Broadcast tá»›i toĂ n bá»™ Client Ä‘ang láº¯ng nghe
        std::lock_guard<std::mutex> lock(g_ws_mutex);
        for (auto* conn_ws : g_ws_connections) {
            conn_ws->send_text(alert_str);
        }
    } catch (const std::exception& e) {
        std::cerr << "[DATABASE ERROR] Loi ghi Event & WebSocket Broadcast: " << e.what() << std::endl;
    }
}

// â”€â”€â”€ AI HELPER FUNCTIONS â”€â”€â”€

// MĂ£ hĂ³a áº£nh OpenCV Mat thĂ nh base64 string Ä‘á»ƒ gá»­i lĂªn OCR service
std::string encodeImageToBase64(const cv::Mat& img) {
    if (img.empty()) return "";
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, 90});
    // Base64 encode thá»§ cĂ´ng (khĂ´ng cáº§n thÆ° viá»‡n ngoĂ i)
    static const char b64chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((buf.size() + 2) / 3) * 4);
    for (size_t i = 0; i < buf.size(); i += 3) {
        uint32_t val = (uint32_t)buf[i] << 16;
        if (i + 1 < buf.size()) val |= (uint32_t)buf[i+1] << 8;
        if (i + 2 < buf.size()) val |= (uint32_t)buf[i+2];
        result += b64chars[(val >> 18) & 0x3F];
        result += b64chars[(val >> 12) & 0x3F];
        result += (i + 1 < buf.size()) ? b64chars[(val >> 6) & 0x3F] : '=';
        result += (i + 2 < buf.size()) ? b64chars[val & 0x3F] : '=';
    }
    return result;
}

// GÂi HTTP POST Ä‘áº¿n OCR service Ä‘á»ƒ Ä‘Âc kĂ½ tá»± biá»ƒn sá»‘
// Tráº£ vÂ ("text", confidence) náº¿u thĂ nh cĂ´ng
std::pair<std::string, float> callOCRService(const cv::Mat& plate_img) {
    if (plate_img.empty()) return {"", 0.0f};
    
    std::string b64 = encodeImageToBase64(plate_img);
    if (b64.empty()) return {"", 0.0f};
    
    std::string json_body = "{\"image_b64\":\"" + b64 + "\"}";
    
    // Táº¡o tá»‡p tmp Ä‘á»ƒ ghi body (trĂ¡nh giá»›i háº¡n command line)
    static std::atomic<std::uint64_t> request_sequence{0};
    const auto request_id = request_sequence.fetch_add(1);
    const auto tmp_file = std::filesystem::temp_directory_path() /
        ("vms_ocr_" + std::to_string(request_id) + ".json");
    struct TemporaryFileGuard {
        std::filesystem::path path;
        ~TemporaryFileGuard() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } temporary_file_guard{tmp_file};
    {
        std::ofstream f(tmp_file);
        if (!f.is_open()) return {"", 0.0f};
        f << json_body;
    }

    // GÂi curl Ä‘á»ƒ POST lĂªn OCR service
#ifdef _WIN32
    std::string cmd = "curl.exe -s --max-time 5 -X POST -H \"Content-Type: application/json\" -d @\"" + tmp_file.string() + "\" http://localhost:8765/ocr";
#else
    std::string cmd = "curl -s --max-time 5 -X POST -H \"Content-Type: application/json\" -d @\"" + tmp_file.string() + "\" http://localhost:8765/ocr";
#endif
    
    std::string response;
    char buf[256];
    FILE* pipe = nullptr;
#ifdef _WIN32
    pipe = _popen(cmd.c_str(), "r");
#else
    pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) {
        return {"", 0.0f};
    }
    while (fgets(buf, sizeof(buf), pipe)) response += buf;
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    // Parse JSON response: {"text":"38A12345","confidence":0.95,...}
    try {
        auto json = crow::json::load(response);
        if (json && json.has("text") && json.has("confidence")) {
            std::string text = json["text"].s();
            float conf = static_cast<float>(json["confidence"].d());
            return {text, conf};
        }
    } catch (...) {}
    return {"", 0.0f};
}

// LÆ°u áº£nh snapshot xuá»‘ng thÆ° má»¥c detections
std::string saveSnapshot(const cv::Mat& img, const std::string& prefix, const std::string& cam_id) {
    if (img.empty()) return "";
    try {
        std::string dir = "snapshots/detections";
        std::filesystem::create_directories(dir);
        
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        time_t t = std::chrono::system_clock::to_time_t(now);
        char ts[32];
        strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", localtime(&t));
        
        static std::atomic<uint64_t> g_snap_counter(0);
        uint64_t cnt = ++g_snap_counter;
        
        std::string filename = dir + "/" + prefix + "_" + cam_id + "_" + ts + "_" + std::to_string(ms.count()) + "_" + std::to_string(cnt) + ".jpg";
        cv::imwrite(filename, img, {cv::IMWRITE_JPEG_QUALITY, 85});
        return filename;
    } catch (...) {
        return "";
    }
}

// Kiá»ƒm tra biá»ƒn sá»‘ Ä‘Ă£ xá»­ lĂ½ gáº§n Ä‘Ă¢y chÆ°a (dedup)
bool isPlateRecentlyProcessed(const std::string& plate_text) {
    if (plate_text.empty()) return false;
    std::lock_guard<std::mutex> lock(g_dedup_mutex);
    time_t now = time(nullptr);
    auto it = g_plate_dedup_cache.find(plate_text);
    if (it != g_plate_dedup_cache.end()) {
        if (now - it->second < PLATE_DEDUP_SECONDS) return true;
        it->second = now;
        return false;
    }
    g_plate_dedup_cache[plate_text] = now;
    return false;
}

// Láº¥y tĂªn vá»‹ trĂ­ camera tá»« cache
std::string getCameraLocation(const std::string& cam_id) {
    std::lock_guard<std::mutex> lock(g_location_mutex);
    auto it = g_camera_location_cache.find(cam_id);
    if (it != g_camera_location_cache.end()) return it->second;
    return "";
}

std::string trimCopy(const std::string& value) {
    const char* ws = " \t\r\n";
    size_t start = value.find_first_not_of(ws);
    if (start == std::string::npos) return "";
    size_t end = value.find_last_not_of(ws);
    return value.substr(start, end - start + 1);
}

std::string runCommandCapture(const std::string& cmd) {
    std::string output;
    char buffer[512];
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return output;
}

struct SystemResourceStats {
    double cpu_percent = -1.0;
    double ram_used_gb = -1.0;
    double ram_total_gb = -1.0;
    double ram_percent = -1.0;
    std::string gpu_name = "N/A";
    double gpu_percent = -1.0;
    double gpu_memory_used_mb = -1.0;
    double gpu_memory_total_mb = -1.0;
};
SystemResourceStats g_cached_resources;
std::mutex g_resources_mutex;

#ifdef _WIN32
unsigned long long fileTimeToULL(const FILETIME& ft) {
    ULARGE_INTEGER uli;
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    return uli.QuadPart;
}
#endif

SystemResourceStats getSystemResourceStats() {
    SystemResourceStats stats;

#ifdef _WIN32
    static unsigned long long prev_idle = 0;
    static unsigned long long prev_kernel = 0;
    static unsigned long long prev_user = 0;
    FILETIME idle_time, kernel_time, user_time;
    if (GetSystemTimes(&idle_time, &kernel_time, &user_time)) {
        unsigned long long idle = fileTimeToULL(idle_time);
        unsigned long long kernel = fileTimeToULL(kernel_time);
        unsigned long long user = fileTimeToULL(user_time);
        if (prev_kernel != 0 || prev_user != 0) {
            unsigned long long sys_delta = (kernel - prev_kernel) + (user - prev_user);
            unsigned long long idle_delta = idle - prev_idle;
            if (sys_delta > 0 && sys_delta >= idle_delta) {
                stats.cpu_percent = 100.0 * static_cast<double>(sys_delta - idle_delta) / static_cast<double>(sys_delta);
            }
        }
        prev_idle = idle;
        prev_kernel = kernel;
        prev_user = user;
    }

    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        const double gb = 1024.0 * 1024.0 * 1024.0;
        stats.ram_total_gb = static_cast<double>(mem.ullTotalPhys) / gb;
        stats.ram_used_gb = static_cast<double>(mem.ullTotalPhys - mem.ullAvailPhys) / gb;
        stats.ram_percent = static_cast<double>(mem.dwMemoryLoad);
    }
#else
    std::ifstream stat_file("/proc/stat");
    std::string cpu_label;
    unsigned long long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0;
    static unsigned long long prev_idle = 0;
    static unsigned long long prev_total = 0;
    if (stat_file >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal) {
        unsigned long long idle_all = idle + iowait;
        unsigned long long total = user + nice + system + idle + iowait + irq + softirq + steal;
        if (prev_total != 0) {
            unsigned long long total_delta = total - prev_total;
            unsigned long long idle_delta = idle_all - prev_idle;
            if (total_delta > 0 && total_delta >= idle_delta) {
                stats.cpu_percent = 100.0 * static_cast<double>(total_delta - idle_delta) / static_cast<double>(total_delta);
            }
        }
        prev_idle = idle_all;
        prev_total = total;
    }

    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    double value = 0.0, total_kb = 0.0, available_kb = 0.0;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemTotal:") total_kb = value;
        if (key == "MemAvailable:") available_kb = value;
    }
    if (total_kb > 0.0) {
        stats.ram_total_gb = total_kb / 1024.0 / 1024.0;
        stats.ram_used_gb = (total_kb - available_kb) / 1024.0 / 1024.0;
        stats.ram_percent = 100.0 * (total_kb - available_kb) / total_kb;
    }
    yaml_file << "streams:\n";
    for (const auto& [cam_id, rtsp_url] : cameras) {
        yaml_file << "  " << cam_id << ": " << rtsp_url << "\n";
        
        // Táº¡o luá»“ng cháº¥t lÆ°á»£ng cao HD (subtype=0) tá»« luá»“ng SD (subtype=1)
        std::string hd_url = rtsp_url;
        size_t pos = hd_url.find("subtype=1");
        if (pos != std::string::npos) {
            hd_url.replace(pos, 9, "subtype=0");
        } else if (hd_url.find("subtype=0") == std::string::npos) {
            // TrÆ°Âng há»£p khĂ´ng cĂ³ tham sá»‘ subtype, thá»­ thĂªm vĂ o
            if (hd_url.find("?") != std::string::npos) {
                hd_url += "&subtype=0";
            } else {
                hd_url += "?subtype=0";
            }
        }
        yaml_file << "  " << cam_id << "_hd: " << hd_url << "\n";
    }

    // Cáº¥u hĂ¬nh cá»•ng WebRTC vĂ  API máº·c Ä‘á»‹nh vá»›i origin * cho CORS
    yaml_file << "\napi:\n  listen: \":1984\"\n  origin: \"*\"\n";
    yaml_file << "\nwebrtc:\n  listen: \":8555\"\n";

    yaml_file.close();
    std::cout << "[VMS SIDE] Da ghi file go2rtc.yaml voi " << cameras.size() << " luong camera." << std::endl;
}

// Khá»Ÿi cháº¡y tiáº¿n trĂ¬nh go2rtc
void startGo2RTCSidecar() {
#ifdef _WIN32
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE; // áº¨n cá»­a sá»• Console cá»§a go2rtc Ä‘á»ƒ giá»¯ mĂ n hĂ¬nh gÂn gĂ ng
    ZeroMemory(&g_go2rtc_pi, sizeof(g_go2rtc_pi));
    std::string cmd = "go2rtc.exe";
    if (!CreateProcessA(NULL, const_cast<char*>(cmd.c_str()), NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &si, &g_go2rtc_pi)) {
        std::cerr << "[VMS SIDE WARNING] Khong the tu dong khoi chay go2rtc.exe. Vui long dam bao tep nay ton tai trong thu muc build/Release." << std::endl;
    } else {
        std::cout << "[VMS SIDE] Da khoi chay go2rtc sidecar engine (PID: " << g_go2rtc_pi.dwProcessId << ")." << std::endl;
    }
#else
    g_go2rtc_pid = fork();
    if (g_go2rtc_pid == 0) {
        execlp("./go2rtc", "./go2rtc", NULL);
        exit(1);
    } else if (g_go2rtc_pid > 0) {
        std::cout << "[VMS SIDE] Da khoi chay go2rtc sidecar engine (PID: " << g_go2rtc_pid << ")." << std::endl;
    }
#endif
}
#endif

    static std::string cached_gpu_name;
    std::string gpu = runCommandCapture("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>nul");
    std::stringstream ss(gpu);
    std::string line;
    if (std::getline(ss, line) && !trimCopy(line).empty()) {
        std::vector<std::string> parts;
        std::stringstream row(line);
        std::string part;
        while (std::getline(row, part, ',')) parts.push_back(trimCopy(part));
        if (parts.size() >= 4) {
            stats.gpu_name = parts[0];
            try {
                stats.gpu_percent = std::stod(parts[1]);
                stats.gpu_memory_used_mb = std::stod(parts[2]);
                stats.gpu_memory_total_mb = std::stod(parts[3]);
            } catch (...) {}
            cached_gpu_name = stats.gpu_name;
            return stats;
        }
    }

    if (cached_gpu_name.empty()) {
#ifdef _WIN32
        std::string wmic = runCommandCapture("wmic path win32_VideoController get Name /value 2>nul");
        std::stringstream wss(wmic);
        while (std::getline(wss, line)) {
            line = trimCopy(line);
            const std::string prefix = "Name=";
            if (line.rfind(prefix, 0) == 0 && line.size() > prefix.size()) {
                cached_gpu_name = trimCopy(line.substr(prefix.size()));
                break;
            }
        }
#else
        cached_gpu_name = trimCopy(runCommandCapture("lspci 2>/dev/null | grep -i 'vga\\|3d\\|display' | head -n 1"));
#endif
    }
    if (!cached_gpu_name.empty()) stats.gpu_name = cached_gpu_name;
    return stats;
}

bool setAICameraState(const std::string& cam_id, const std::string& name, const std::string& stream_id, bool enabled) {
    if (cam_id.empty()) return false;
    {
        std::lock_guard<std::mutex> lock(g_ai_list_mutex);
        if (enabled) {
            g_ai_camera_list.insert(cam_id);
        } else {
            g_ai_camera_list.erase(cam_id);
        }
    }

    try {
        pqxx::connection conn(DB_CONN_STR);
        pqxx::work txn(conn);
        txn.exec("UPDATE cameras SET ai_enabled = " + std::string(enabled ? "true" : "false") +
                 " WHERE camera_id = " + txn.quote(cam_id));
        txn.commit();
    } catch (const std::exception& error) {
        std::cerr << "[AI STATE WARNING] Could not persist " << cam_id << ": " << error.what() << '\n';
    }

    const std::string effective_stream_id = stream_id.empty() ? cam_id : stream_id;
    if (!enabled) {
        std::unique_ptr<RTSPCapture> retired_capture;
        {
            std::lock_guard<std::mutex> lock(g_cameras_mutex);
            if (g_runtime_camera_ids.find(cam_id) != g_runtime_camera_ids.end()) {
                return true;
            }
            const auto camera = std::find_if(
                g_active_cameras.begin(), g_active_cameras.end(),
                [&cam_id](const ActiveCameraInfo& item) {
                    return item.camera_id == cam_id;
                });
            if (camera != g_active_cameras.end()) {
                retired_capture = std::move(camera->capture);
                g_active_cameras.erase(camera);
            }
        }
        retireCapture(cam_id, std::move(retired_capture), true);
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(g_cameras_mutex);
        const auto existing = std::find_if(
            g_active_cameras.begin(), g_active_cameras.end(),
            [&cam_id](const ActiveCameraInfo& item) {
                return item.camera_id == cam_id;
            });
        if (existing != g_active_cameras.end() && existing->capture &&
            existing->stream_id == effective_stream_id) {
            return true;
        }
    }

    const std::string local_rtsp = "rtsp://127.0.0.1:8554/" + effective_stream_id;
    auto replacement = std::make_unique<RTSPCapture>(cam_id, local_rtsp);
    if (!replacement->start()) return false;

    std::unique_ptr<RTSPCapture> retired_capture;
    {
        std::lock_guard<std::mutex> lock(g_cameras_mutex);
        const auto existing = std::find_if(
            g_active_cameras.begin(), g_active_cameras.end(),
            [&cam_id](const ActiveCameraInfo& item) {
                return item.camera_id == cam_id;
            });
        if (existing != g_active_cameras.end()) {
            if (existing->capture && existing->stream_id == effective_stream_id) {
                retired_capture = std::move(replacement);
            } else {
                retired_capture = std::move(existing->capture);
                existing->capture = std::move(replacement);
                existing->stream_id = effective_stream_id;
            }
        } else {
            g_active_cameras.push_back(
                {cam_id, name, std::move(replacement), effective_stream_id});
        }
    }
    retireCapture(cam_id, std::move(retired_capture), false);
    return true;
}

// LÆ°u káº¿t quáº£ phĂ¡t hiá»‡n vĂ o báº£ng detections
void saveDetection(
    const std::string& camera_id,
    const std::string& object_type,
    float object_confidence,
    const std::string& plate_text,
    float plate_conf,
    const std::string& identity_label,
    float identity_conf,
    const std::string& vehicle_img_path,
    const std::string& plate_img_path,
    const std::string& face_img_path,
    int bx, int by, int bw, int bh,
    const std::string& location,
    bool emit_object_event = true
) {
    try {
        pqxx::connection conn(DB_CONN_STR);
        pqxx::work txn(conn);
        conn.prepare("ins_det",
            "INSERT INTO detections "
            "(camera_id, object_type, plate_text, plate_confidence, identity_label, identity_confidence,"
            " vehicle_image_path, plate_image_path, face_image_path,"
            " bbox_x, bbox_y, bbox_w, bbox_h, location_name)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)"
        );
        txn.exec_prepared("ins_det",
            camera_id, object_type, plate_text, plate_conf,
            identity_label, identity_conf,
            vehicle_img_path, plate_img_path, face_img_path,
            bx, by, bw, bh, location
        );
        txn.commit();
    } catch (const std::exception& e) {
        std::cerr << "[DETECTION SAVE ERROR] " << e.what() << std::endl;
    }

    // Âá»“ng bá»™ sang báº£ng events Ä‘á»ƒ hiá»ƒn thá»‹ lĂªn cĂ¡c tab tÆ°Æ¡ng á»©ng cá»§a giao diá»‡n Web
    try {
        static const std::string kPersonDescription =
            "Ng" "\xC6\xB0" "\xE1\xBB\x9D" "i " "\xC4\x91" "i b" "\xE1\xBB\x99";
        static const std::string kVehicleDescription =
            "Ph" "\xC3\xA1" "t hi" "\xE1\xBB\x87" "n ph" "\xC6\xB0" "\xC6\xA1" "ng ti" "\xE1\xBB\x87" "n";
        static const std::string kPlatePrefix =
            "Bi" "\xE1\xBB\x83" "n s" "\xE1\xBB\x91" " xe: ";

        if (object_type == "person" && emit_object_event) {
            saveEvent(camera_id, "person_detected", kPersonDescription, object_confidence, vehicle_img_path);
            if (!identity_label.empty()) {
                std::string desc = "Face matched: " + identity_label + " (Similarity: " + std::to_string(static_cast<int>(identity_conf * 100)) + "%)";
                saveEvent(camera_id, "face_recognized", desc, identity_conf, face_img_path.empty() ? vehicle_img_path : face_img_path);
            }
        } else if (object_type.rfind("vehicle_", 0) == 0) {
            const std::string desc = plate_text.empty()
                ? kVehicleDescription
                : kPlatePrefix + plate_text;
            const std::string effective_plate_path =
                plate_img_path.empty() ? vehicle_img_path : plate_img_path;
            if (emit_object_event) {
                saveEvent(camera_id, "vehicle_detected", desc,
                          plate_text.empty() ? object_confidence : plate_conf,
                          vehicle_img_path, plate_text,
                          plate_text.empty() ? "" : effective_plate_path);
            }
            if (!plate_text.empty()) {
                saveEvent(camera_id, "plate_recognized", kPlatePrefix + plate_text,
                          plate_conf, effective_plate_path, plate_text,
                          effective_plate_path);
            }
        }    } catch (const std::exception& e) {
        std::cerr << "[DETECTION TO EVENT ERROR] " << e.what() << std::endl;
    }
}

// YOLOv11 COCO class names (chá»‰ lÂc cĂ¡c class cáº§n thiáº¿t)
static const std::vector<std::string> COCO_CLASSES = {
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

// â”€â”€â”€ THREAD CHáº Y Xá»¬ LÂ AI BACKGROUND â”€â”€â”€


// --- ROI AND THRESHOLDS ---
struct AiRoiConfig {
    bool enabled = false;
    std::vector<cv::Point2f> points; // Relative coordinates 0.0 - 1.0
};
std::mutex g_roi_mutex;
std::unordered_map<std::string, AiRoiConfig> g_ai_roi_cache;

AiRoiConfig getAiRoiConfig(const std::string& cam_id) {
    std::lock_guard<std::mutex> lock(g_roi_mutex);
    auto it = g_ai_roi_cache.find(cam_id);
    if (it != g_ai_roi_cache.end()) return it->second;
    return {false, {}};
}

cv::Rect aiRoiToRect(const AiRoiConfig& roi, const cv::Size& frame_size) {
    if (!roi.enabled || roi.points.empty()) return cv::Rect(0, 0, frame_size.width, frame_size.height);
    float min_x = 1.0f, min_y = 1.0f, max_x = 0.0f, max_y = 0.0f;
    for (const auto& p : roi.points) {
        min_x = std::min(min_x, p.x); min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x); max_y = std::max(max_y, p.y);
    }
    int x = std::clamp(static_cast<int>(min_x * frame_size.width), 0, frame_size.width - 1);
    int y = std::clamp(static_cast<int>(min_y * frame_size.height), 0, frame_size.height - 1);
    int w = std::clamp(static_cast<int>((max_x - min_x) * frame_size.width), 1, frame_size.width - x);
    int h = std::clamp(static_cast<int>((max_y - min_y) * frame_size.height), 1, frame_size.height - y);
    return cv::Rect(x, y, w, h);
}

bool buildAiRoiFrame(const std::string& cam_id, const cv::Mat& frame, cv::Mat& out_frame, cv::Point& out_offset) {
    if (frame.empty()) return false;
    AiRoiConfig roi = getAiRoiConfig(cam_id);
    cv::Rect rect = aiRoiToRect(roi, frame.size());
    if (rect.empty()) return false;

    out_frame = frame(rect).clone();
    out_offset = rect.tl();
    if (roi.enabled && roi.points.size() >= 3) {
        cv::Mat mask = cv::Mat::zeros(out_frame.size(), CV_8UC1);
        std::vector<cv::Point> pts;
        for (const auto& p : roi.points) {
            pts.push_back(cv::Point(std::round(p.x * frame.cols) - rect.x, std::round(p.y * frame.rows) - rect.y));
        }
        std::vector<std::vector<cv::Point>> polys = {pts};
        cv::fillPoly(mask, polys, cv::Scalar(255));
        cv::Mat masked;
        out_frame.copyTo(masked, mask);
        out_frame = masked;
    }
    return true;
}

std::atomic<float> g_vehicle_det_threshold(0.25f);
std::atomic<float> g_plate_det_threshold(0.20f);
std::atomic<float> g_face_det_threshold(0.50f);
std::atomic<float> g_plate_ocr_threshold{0.45F};
std::atomic<float> g_face_match_threshold{0.36F};

bool shouldRunObjectDetection(const std::string& camera_id) {
    static std::mutex cadence_mutex;
    static std::unordered_map<std::string, std::uint32_t> cadence;
    std::lock_guard<std::mutex> lock(cadence_mutex);
    auto& frame_count = cadence[camera_id];
    const bool should_run = frame_count % 3 == 0;
    ++frame_count;
    return should_run;
}


// --- TRACKING & DEDUPLICATION ---
struct ObjectCaptureTrack {
    std::string track_id;
    std::string camera_id;
    std::string object_type;
    cv::Rect last_box;
    std::chrono::steady_clock::time_point last_seen{};
    std::uint64_t last_frame_token = 0;
    int seen_count = 0;
    float best_score = 0.0f;
    int plate_attempts = 0;
    std::chrono::steady_clock::time_point last_plate_attempt{};
    bool plate_recognized = false;

    struct Candidate {
        float score;
        cv::Mat object_crop;
        cv::Mat face_crop;
        cv::Mat plate_crop;
        std::string plate_text;
        float plate_conf;
        std::string face_label;
        float face_conf;
    };
    std::vector<Candidate> candidates;
    int best_index = 0;
};

struct ObjectCaptureGroup {
    std::string track_id;
    std::vector<ObjectCaptureTrack::Candidate> candidates;
    int best_index = 0;
};

struct ObjectTrackObservation {
    std::string track_id;
    std::optional<ObjectCaptureGroup> new_capture;
    bool should_analyze_plate = false;
    bool is_plate_retry = false;
};

std::mutex g_tracking_mutex;
std::unordered_map<std::string, ObjectCaptureTrack> g_tracking_cache;
std::atomic<std::uint64_t> g_next_tracking_id{1};
std::atomic<std::uint64_t> g_next_ai_frame_token{1};

float boxIntersectionOverUnion(const cv::Rect& left, const cv::Rect& right) {
    const auto intersection = left & right;
    const float intersection_area = static_cast<float>(intersection.area());
    const float union_area =
        static_cast<float>(left.area() + right.area()) - intersection_area;
    return union_area > 0.0F ? intersection_area / union_area : 0.0F;
}

ObjectTrackObservation collectTrackedObjectCapture(
    const std::string& cam_id,
    const std::string& object_type,
    const cv::Rect& box,
    const cv::Mat& crop,
    float det_conf,
    std::uint64_t frame_token
) {
    std::lock_guard<std::mutex> lock(g_tracking_mutex);
    const auto now = std::chrono::steady_clock::now();

    for (auto iterator = g_tracking_cache.begin();
         iterator != g_tracking_cache.end();) {
        if (now - iterator->second.last_seen > std::chrono::seconds{4}) {
            iterator = g_tracking_cache.erase(iterator);
        } else {
            ++iterator;
        }
    }

    ObjectCaptureTrack* matched_track = nullptr;
    float best_match_score = -1.0F;
    const cv::Point2f box_center{
        box.x + box.width * 0.5F, box.y + box.height * 0.5F};

    for (auto& [track_id, track] : g_tracking_cache) {
        if (track.camera_id != cam_id || track.object_type != object_type ||
            track.last_frame_token == frame_token) {
            continue;
        }

        const float iou = boxIntersectionOverUnion(track.last_box, box);
        const cv::Point2f track_center{
            track.last_box.x + track.last_box.width * 0.5F,
            track.last_box.y + track.last_box.height * 0.5F};
        const float center_distance = cv::norm(box_center - track_center);
        const float reference_size = std::max(
            1.0F,
            0.5F * (std::hypot(static_cast<float>(box.width),
                               static_cast<float>(box.height)) +
                    std::hypot(static_cast<float>(track.last_box.width),
                               static_cast<float>(track.last_box.height))));
        const float normalized_distance = center_distance / reference_size;
        const float larger_area =
            static_cast<float>(std::max(box.area(), track.last_box.area()));
        const float size_ratio = larger_area > 0.0F
            ? static_cast<float>(std::min(box.area(), track.last_box.area())) /
                  larger_area
            : 0.0F;

        if (iou < 0.10F &&
            (normalized_distance > 2.0F || size_ratio < 0.20F)) {
            continue;
        }

        const float match_score =
            iou * 2.0F + std::max(0.0F, 1.0F - normalized_distance) +
            size_ratio * 0.25F;
        if (match_score > best_match_score) {
            best_match_score = match_score;
            matched_track = &track;
        }
    }

    const bool is_new_track = matched_track == nullptr;
    if (is_new_track) {
        const auto sequence = g_next_tracking_id.fetch_add(1);
        const std::string track_id =
            cam_id + "|" + object_type + "|" + std::to_string(sequence);
        auto iterator = g_tracking_cache.try_emplace(track_id).first;
        matched_track = &iterator->second;
        matched_track->track_id = track_id;
        matched_track->camera_id = cam_id;
        matched_track->object_type = object_type;
    }

    auto& track = *matched_track;
    track.seen_count++;
    track.last_seen = now;
    track.last_box = box;
    track.last_frame_token = frame_token;

    ObjectTrackObservation observation;
    observation.track_id = track.track_id;
    if (object_type == "vehicle" && !track.plate_recognized &&
        track.plate_attempts < 4 &&
        (track.plate_attempts == 0 ||
         now - track.last_plate_attempt >= std::chrono::milliseconds{600})) {
        observation.should_analyze_plate = true;
        observation.is_plate_retry = !is_new_track;
        track.plate_attempts++;
        track.last_plate_attempt = now;
    }

    if (!is_new_track) return observation;

    ObjectCaptureTrack::Candidate candidate;
    candidate.score = det_conf;
    if (!crop.empty()) candidate.object_crop = crop.clone();
    candidate.plate_conf = 0.0F;
    candidate.face_conf = 0.0F;
    track.best_score = det_conf;
    track.candidates.push_back(std::move(candidate));

    ObjectCaptureGroup group;
    group.track_id = track.track_id;
    group.candidates = track.candidates;
    group.best_index = 0;
    observation.new_capture = std::move(group);
    return observation;
}

void markTrackPlateRecognized(const std::string& track_id) {
    std::lock_guard<std::mutex> lock(g_tracking_mutex);
    const auto iterator = g_tracking_cache.find(track_id);
    if (iterator != g_tracking_cache.end()) {
        iterator->second.plate_recognized = true;
    }
}

struct SavedCaptureGroup {
    std::string object_path;
    std::string plate_path;
    std::string face_path;
    std::string extra_data;
};

SavedCaptureGroup saveCaptureGroupImages(
    const ObjectCaptureGroup& group,
    const std::string& cam_id,
    const std::string& object_prefix
) {
    SavedCaptureGroup saved;
    if (group.candidates.empty()) return saved;
    int best_index = std::clamp(group.best_index, 0, static_cast<int>(group.candidates.size()) - 1);
    
    const auto& best = group.candidates[best_index];
    saved.object_path = saveSnapshot(best.object_crop, object_prefix, cam_id);
    if (!best.plate_crop.empty()) saved.plate_path = saveSnapshot(best.plate_crop, "plate", cam_id);
    if (!best.face_crop.empty()) saved.face_path = saveSnapshot(best.face_crop, "face", cam_id);
    
    return saved;
}

cv::Rect expandRect(const cv::Rect& rect, const cv::Size& bounds, float pad_x, float pad_y, int min_pad = 0) {
    int exp_w = std::max(static_cast<int>(rect.width * pad_x), min_pad);
    int exp_h = std::max(static_cast<int>(rect.height * pad_y), min_pad);
    int cx = rect.x + rect.width / 2;
    int cy = rect.y + rect.height / 2;
    int new_w = rect.width + exp_w * 2;
    int new_h = rect.height + exp_h * 2;
    int new_x = cx - new_w / 2;
    int new_y = cy - new_h / 2;
    cv::Rect expanded(new_x, new_y, new_w, new_h);
    
    int x = std::max(0, expanded.x);
    int y = std::max(0, expanded.y);
    int w = std::min(expanded.width, bounds.width - x);
    int h = std::min(expanded.height, bounds.height - y);
    if (w <= 0 || h <= 0) return cv::Rect();
    return cv::Rect(x, y, w, h);
}

cv::Rect objectContextRect(const cv::Rect& rect, const cv::Size& bounds, bool is_person) {
    float pad_x = is_person ? 0.75f : 0.55f;
    float pad_y = is_person ? 0.55f : 0.70f;
    int min_pad = is_person ? 90 : 130;
    return expandRect(rect, bounds, pad_x, pad_y, min_pad);
}

struct MotionProbeResult {
    bool detected{false};
    cv::Rect source_roi;
    double changed_ratio{0.0};
};

struct CameraMotionState {
    cv::Mat previous_gray;
    std::chrono::steady_clock::time_point last_motion{};
    cv::Rect last_motion_bounds;
};

MotionProbeResult detectMotionForAi(
    CameraMotionState& state,
    const cv::Mat& frame,
    std::chrono::steady_clock::time_point now
) {
    MotionProbeResult result;
    if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) return result;

    constexpr int probe_width{320};
    constexpr int binary_threshold{20};
    constexpr double minimum_changed_ratio{0.0012};
    constexpr auto motion_hold{std::chrono::milliseconds{500}};

    const int probe_height = std::max(
        1, static_cast<int>(
               std::lround(static_cast<double>(frame.rows) * probe_width /
                           static_cast<double>(frame.cols))));
    cv::Mat resized;
    cv::resize(
        frame, resized, cv::Size{probe_width, probe_height},
        0.0, 0.0, cv::INTER_AREA);

    cv::Mat gray;
    if (resized.channels() == 1) {
        gray = resized;
    } else if (resized.channels() == 4) {
        cv::cvtColor(resized, gray, cv::COLOR_BGRA2GRAY);
    } else {
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    }
    cv::GaussianBlur(gray, gray, cv::Size{5, 5}, 0.0);

    if (state.previous_gray.empty() ||
        state.previous_gray.size() != gray.size()) {
        state.previous_gray = gray.clone();
        state.last_motion = now;
        state.last_motion_bounds =
            cv::Rect{0, 0, gray.cols, gray.rows};
        result.detected = true;
        result.source_roi = cv::Rect{0, 0, frame.cols, frame.rows};
        return result;
    }

    cv::Mat difference;
    cv::absdiff(state.previous_gray, gray, difference);
    state.previous_gray = gray.clone();

    cv::Mat motion_mask;
    cv::threshold(
        difference, motion_mask, binary_threshold, 255, cv::THRESH_BINARY);

    // Camera timestamps usually change in the upper corners every second.
    const int overlay_height = std::max(8, motion_mask.rows / 10);
    const int overlay_width = motion_mask.cols * 45 / 100;
    cv::rectangle(
        motion_mask, cv::Rect{0, 0, overlay_width, overlay_height},
        cv::Scalar{0}, cv::FILLED);
    cv::rectangle(
        motion_mask,
        cv::Rect{
            motion_mask.cols - overlay_width, 0,
            overlay_width, overlay_height},
        cv::Scalar{0}, cv::FILLED);

    const cv::Mat open_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size{3, 3});
    const cv::Mat dilate_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size{5, 5});
    cv::morphologyEx(
        motion_mask, motion_mask, cv::MORPH_OPEN, open_kernel);
    cv::dilate(motion_mask, motion_mask, dilate_kernel, cv::Point{-1, -1}, 2);

    result.changed_ratio =
        static_cast<double>(cv::countNonZero(motion_mask)) /
        static_cast<double>(motion_mask.total());

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        motion_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    const double minimum_contour_area = std::max(
        48.0, static_cast<double>(motion_mask.total()) * 0.0008);

    cv::Rect motion_bounds;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < minimum_contour_area) continue;
        const cv::Rect contour_bounds = cv::boundingRect(contour);
        motion_bounds = motion_bounds.empty()
            ? contour_bounds
            : (motion_bounds | contour_bounds);
    }

    const bool direct_motion =
        !motion_bounds.empty() &&
        result.changed_ratio >= minimum_changed_ratio;
    if (direct_motion) {
        state.last_motion = now;
        state.last_motion_bounds = motion_bounds;
    } else if (!state.last_motion_bounds.empty() &&
               now - state.last_motion <= motion_hold) {
        motion_bounds = state.last_motion_bounds;
    } else {
        return result;
    }

    const double scale_x =
        static_cast<double>(frame.cols) / static_cast<double>(gray.cols);
    const double scale_y =
        static_cast<double>(frame.rows) / static_cast<double>(gray.rows);
    const int source_x =
        std::clamp(static_cast<int>(std::floor(motion_bounds.x * scale_x)),
                   0, frame.cols - 1);
    const int source_y =
        std::clamp(static_cast<int>(std::floor(motion_bounds.y * scale_y)),
                   0, frame.rows - 1);
    const int source_right = std::clamp(
        static_cast<int>(
            std::ceil((motion_bounds.x + motion_bounds.width) * scale_x)),
        source_x + 1, frame.cols);
    const int source_bottom = std::clamp(
        static_cast<int>(
            std::ceil((motion_bounds.y + motion_bounds.height) * scale_y)),
        source_y + 1, frame.rows);

    const cv::Rect raw_source_roi{
        source_x, source_y,
        source_right - source_x, source_bottom - source_y};
    result.source_roi = expandRect(
        raw_source_roi, frame.size(), 0.80F, 0.80F, 48);
    if (result.source_roi.empty()) {
        result.source_roi = cv::Rect{0, 0, frame.cols, frame.rows};
    }
    result.detected = true;
    return result;
}

// --- NEW CONCURRENT AI WORKER ---
void processCameraAI(
    const std::string& cam_id,
    cv::Mat frame,
    const std::optional<cv::Rect>& motion_source_roi
) {
    if (frame.empty()) return;

    cv::Mat ai_frame;
    cv::Point ai_offset(0, 0);
    if (!buildAiRoiFrame(cam_id, frame, ai_frame, ai_offset)) {
        ai_frame = frame;
    }

    if (motion_source_roi && !motion_source_roi->empty()) {
        const cv::Rect local_motion_roi{
            motion_source_roi->x - ai_offset.x,
            motion_source_roi->y - ai_offset.y,
            motion_source_roi->width,
            motion_source_roi->height};
        const cv::Rect clipped_motion_roi =
            local_motion_roi & cv::Rect{0, 0, ai_frame.cols, ai_frame.rows};
        const auto full_area =
            static_cast<std::int64_t>(ai_frame.cols) * ai_frame.rows;
        if (!clipped_motion_roi.empty() &&
            static_cast<std::int64_t>(clipped_motion_roi.area()) <
                full_area * 9 / 10) {
            ai_frame = ai_frame(clipped_motion_roi);
            ai_offset += clipped_motion_roi.tl();
        }
    }

    const auto frame_token = g_next_ai_frame_token.fetch_add(1);
    std::vector<DetectionBox> frame_boxes;
    g_ai_stats.frames_processed++;
    const std::string location = getCameraLocation(cam_id);

    // Plate-first pipeline: one detector pass on the newest motion ROI,
    // independent from the slower object tracking/event path.
    if (g_ai_plate_enabled.load() && g_plate_detector) {
        g_ai_stats.plate_attempts++;
        const auto plate_detections = g_plate_detector->detect(
            ai_frame, g_plate_det_threshold.load(), 0.35F);
        for (const auto& plate_detection : plate_detections) {
            const cv::Rect plate_box =
                plate_detection.box &
                cv::Rect{0, 0, ai_frame.cols, ai_frame.rows};
            if (plate_box.width < 10 || plate_box.height < 5) continue;
            const float aspect_ratio =
                static_cast<float>(plate_box.width) /
                static_cast<float>(plate_box.height);
            if (aspect_ratio < 0.65F || aspect_ratio > 7.0F) continue;

            const cv::Rect source_plate_box{
                plate_box.x + ai_offset.x,
                plate_box.y + ai_offset.y,
                plate_box.width,
                plate_box.height};
            if (source_plate_box.y < frame.rows / 10) continue;

            g_ai_stats.plate_boxes_detected++;
            cv::Rect expanded_plate = expandRect(
                plate_box, ai_frame.size(), 0.20F, 0.35F, 2);
            if (expanded_plate.empty()) expanded_plate = plate_box;
            const cv::Mat plate_capture =
                ai_frame(expanded_plate).clone();
            const auto ocr_result = callOCRService(plate_capture);
            if (ocr_result.first.empty() ||
                ocr_result.second < g_plate_ocr_threshold.load()) {
                continue;
            }

            frame_boxes.push_back({
                source_plate_box.x, source_plate_box.y,
                source_plate_box.width, source_plate_box.height,
                ocr_result.first, ocr_result.second});
            if (isPlateRecentlyProcessed(ocr_result.first)) continue;

            cv::Rect vehicle_context = expandRect(
                plate_box, ai_frame.size(), 2.5F, 3.0F, 64);
            if (vehicle_context.empty()) vehicle_context = expanded_plate;
            const std::string vehicle_path = saveSnapshot(
                ai_frame(vehicle_context), "vehicle_plate", cam_id);
            const std::string plate_path = saveSnapshot(
                plate_capture, "plate", cam_id);
            g_ai_stats.total_plates++;
            saveDetection(
                cam_id, "vehicle_plate", plate_detection.confidence,
                ocr_result.first, ocr_result.second, "", 0.0F,
                vehicle_path, plate_path, "",
                source_plate_box.x, source_plate_box.y,
                source_plate_box.width, source_plate_box.height,
                location, false);
        }
    }

    if (g_vehicle_detector && g_ai_object_enabled.load() &&
        shouldRunObjectDetection(cam_id)) {
        const auto detections = g_vehicle_detector->detect(
            ai_frame, g_vehicle_det_threshold.load(), 0.45F);

        for (const auto& detection : detections) {
            const int class_id = detection.class_id;
            if (class_id != 0 && class_id != 2 && class_id != 3 &&
                class_id != 5 && class_id != 7) {
                continue;
            }

            std::string object_type;
            if (class_id == 0) object_type = "person";
            else if (class_id == 2) object_type = "vehicle_car";
            else if (class_id == 3) object_type = "vehicle_motorbike";
            else if (class_id == 5) object_type = "vehicle_bus";
            else object_type = "vehicle_truck";

            const cv::Rect object_box =
                detection.box & cv::Rect(0, 0, ai_frame.cols, ai_frame.rows);
            if (object_box.width < 18 || object_box.height < 18) continue;

            const cv::Rect source_box(
                object_box.x + ai_offset.x, object_box.y + ai_offset.y,
                object_box.width, object_box.height);
            const cv::Rect context_box =
                objectContextRect(object_box, ai_frame.size(), class_id == 0);
            const cv::Mat object_crop = ai_frame(context_box);
            if (object_crop.empty()) continue;

            const std::string base_label =
                class_id == 0 ? "person" : object_type.substr(8);
            frame_boxes.push_back({
                source_box.x, source_box.y, source_box.width, source_box.height,
                base_label, detection.confidence});

            const std::string tracking_type =
                class_id == 0 ? "person" : "vehicle";
            auto observation = collectTrackedObjectCapture(
                cam_id, tracking_type, source_box, object_crop,
                detection.confidence, frame_token);
            if (class_id == 0) {
                g_ai_stats.total_persons++;
            } else {
                g_ai_stats.total_vehicles++;
            }

            if (class_id == 0) {
                if (!observation.new_capture) continue;
                auto& group = *observation.new_capture;
                auto& candidate = group.candidates[group.best_index];
                std::string face_label;
                float face_confidence = 0.0F;
                cv::Mat face_capture;

                if (g_ai_face_enabled.load() && g_face_detector &&
                    g_face_embedder) {
                    const auto face_detections = g_face_detector->detect(
                        object_crop, g_face_det_threshold.load());
                    for (const auto& face_detection : face_detections) {
                        const cv::Rect face_box =
                            face_detection.box &
                            cv::Rect(0, 0, object_crop.cols, object_crop.rows);
                        if (face_box.width < 15 || face_box.height < 15) continue;

                        face_capture = object_crop(face_box);
                        const auto embedding =
                            g_face_embedder->extractEmbedding(face_capture);
                        float maximum_similarity = 0.0F;
                        std::string matched_label;
                        {
                            std::lock_guard<std::mutex> lock(g_identities_mutex);
                            for (const auto& identity : g_face_identities_cache) {
                                const float similarity =
                                    FaceEmbedder::calculateSimilarity(
                                        embedding, identity.embedding);
                                if (similarity > maximum_similarity) {
                                    maximum_similarity = similarity;
                                    matched_label = identity.label;
                                }
                            }
                        }
                        if (maximum_similarity >=
                            g_face_match_threshold.load()) {
                            g_ai_stats.total_faces++;
                            face_label = matched_label;
                            face_confidence = maximum_similarity;
                        }
                        break;
                    }
                }

                if (!face_capture.empty()) {
                    candidate.face_crop = face_capture.clone();
                }
                candidate.face_label = face_label;
                candidate.face_conf = face_confidence;
                if (!face_label.empty()) {
                    frame_boxes.back().label = face_label;
                    frame_boxes.back().confidence = face_confidence;
                }

                const SavedCaptureGroup saved =
                    saveCaptureGroupImages(group, cam_id, "person");
                saveDetection(
                    cam_id, object_type, detection.confidence, "", 0.0F,
                    face_label, face_confidence, saved.object_path, "",
                    saved.face_path, source_box.x, source_box.y,
                    source_box.width, source_box.height, location);
                continue;
            }

            if (!observation.new_capture) continue;
            const auto& group = *observation.new_capture;
            const SavedCaptureGroup saved =
                saveCaptureGroupImages(group, cam_id, "vehicle");
            saveDetection(
                cam_id, object_type, detection.confidence, "", 0.0F,
                "", 0.0F, saved.object_path, "", "",
                source_box.x, source_box.y,
                source_box.width, source_box.height, location);
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_detections_mutex);
        g_detections_cache[cam_id] = {
            cam_id, frame_boxes, time(nullptr), frame.cols, frame.rows};
    }
}

void aiWorkerLoop() {
    const auto worker_count = [] {
        if (const char* configured = std::getenv("VMS_AI_WORKERS")) {
            try {
                return std::clamp(
                    static_cast<std::size_t>(std::stoul(configured)),
                    std::size_t{1}, std::size_t{4});
            } catch (...) {
            }
        }
        return std::size_t{2};
    }();
    const auto sample_interval = [] {
        if (const char* configured = std::getenv("VMS_AI_SAMPLE_MS")) {
            try {
                return std::chrono::milliseconds{
                    std::clamp(std::stol(configured), 150L, 5000L)};
            } catch (...) {
            }
        }
        return std::chrono::milliseconds{250};
    }();

    struct AIJob {
        std::string camera_id;
        RTSPCapture::FramePtr frame;
        std::optional<cv::Rect> motion_source_roi;
    };

    std::deque<AIJob> jobs;
    std::unordered_set<std::string> busy_cameras;
    std::unordered_map<
        std::string, std::chrono::steady_clock::time_point> last_scheduled;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool stopping{false};
    const auto queue_limit = worker_count * 2;

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index) {
        workers.emplace_back([&] {
            while (true) {
                AIJob job;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(
                        lock, [&] { return stopping || !jobs.empty(); });
                    if (stopping && jobs.empty()) return;
                    job = std::move(jobs.front());
                    jobs.pop_front();
                }
                if (job.frame && !job.frame->empty()) {
                    processCameraAI(
                        job.camera_id, *job.frame, job.motion_source_roi);
                }
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    busy_cameras.erase(job.camera_id);
                }
            }
        });
    }

    std::cout << "[AI WORKER] " << worker_count
              << " workers, motion-first sampling "
              << sample_interval.count() << " ms/camera.\n";

    std::unordered_map<std::string, CameraMotionState> motion_states;
    std::unordered_map<std::string, RTSPCapture::FramePtr>
        last_checked_frames;
    std::unordered_map<
        std::string, std::chrono::steady_clock::time_point>
        last_motion_probe;
    std::unordered_map<std::string, AIJob> pending_motion_jobs;
    std::unordered_map<
        std::string, std::chrono::steady_clock::time_point>
        last_cache_heartbeat;

    std::size_t round_robin_offset{0};
    while (g_running.load()) {
        if (!g_ai_enabled.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
            continue;
        }

        std::vector<std::string> ai_cameras;
        {
            std::lock_guard<std::mutex> lock(g_ai_list_mutex);
            ai_cameras.assign(
                g_ai_camera_list.begin(), g_ai_camera_list.end());
        }
        std::sort(ai_cameras.begin(), ai_cameras.end());
        if (ai_cameras.empty()) {
            round_robin_offset = 0;
            pending_motion_jobs.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        for (std::size_t checked = 0; checked < ai_cameras.size(); ++checked) {
            const auto camera_index =
                (round_robin_offset + checked) % ai_cameras.size();
            const auto& camera_id = ai_cameras[camera_index];

            RTSPCapture::FramePtr frame;
            {
                std::lock_guard<std::mutex> lock(g_cameras_mutex);
                const auto camera = std::find_if(
                    g_active_cameras.begin(), g_active_cameras.end(),
                    [&camera_id](const ActiveCameraInfo& item) {
                        return item.camera_id == camera_id;
                    });
                if (camera != g_active_cameras.end() && camera->capture &&
                    camera->capture->isOpened()) {
                    frame = camera->capture->getLatestFrameShared();
                }
            }

            const bool motion_enabled = g_ai_motion_enabled.load();
            const auto previous_probe = last_motion_probe.find(camera_id);
            const bool motion_probe_due =
                previous_probe == last_motion_probe.end() ||
                now - previous_probe->second >=
                    std::chrono::milliseconds{120};
            if (frame && !frame->empty() &&
                last_checked_frames[camera_id].get() != frame.get() &&
                (!motion_enabled || motion_probe_due)) {
                last_checked_frames[camera_id] = frame;

                if (motion_enabled) {
                    last_motion_probe[camera_id] = now;
                    g_ai_stats.motion_frames_checked++;
                    const auto motion = detectMotionForAi(
                        motion_states[camera_id], *frame, now);
                    if (motion.detected) {
                        g_ai_stats.motion_frames_triggered++;
                        pending_motion_jobs.insert_or_assign(
                            camera_id,
                            AIJob{camera_id, frame, motion.source_roi});
                    } else {
                        g_ai_stats.motion_frames_skipped++;
                        const auto heartbeat =
                            last_cache_heartbeat.find(camera_id);
                        if (heartbeat == last_cache_heartbeat.end() ||
                            now - heartbeat->second >=
                                std::chrono::seconds{2}) {
                            {
                                std::lock_guard<std::mutex> lock(
                                    g_detections_mutex);
                                g_detections_cache[camera_id] = {
                                    camera_id, {}, time(nullptr),
                                    frame->cols, frame->rows};
                            }
                            last_cache_heartbeat[camera_id] = now;
                        }
                    }
                } else {
                    pending_motion_jobs.insert_or_assign(
                        camera_id,
                        AIJob{camera_id, frame, std::nullopt});
                }
            }

            const auto pending = pending_motion_jobs.find(camera_id);
            if (pending == pending_motion_jobs.end()) continue;

            bool queued{false};
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                const auto last = last_scheduled.find(camera_id);
                if (jobs.size() < queue_limit &&
                    !busy_cameras.contains(camera_id) &&
                    (last == last_scheduled.end() ||
                     now - last->second >= sample_interval)) {
                    busy_cameras.insert(camera_id);
                    last_scheduled[camera_id] = now;
                    jobs.push_back(std::move(pending->second));
                    queued = true;
                }
            }
            if (queued) {
                pending_motion_jobs.erase(pending);
                g_ai_stats.motion_frames_queued++;
                queue_cv.notify_one();
            }
        }

        round_robin_offset =
            (round_robin_offset + 1) % ai_cameras.size();
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stopping = true;
        jobs.clear();
    }
    queue_cv.notify_all();
    for (auto& worker : workers) {
        if (worker.joinable()) worker.join();
    }
    std::cout << "[AI WORKER] Da dung.\n";
}

#ifdef _WIN32
PROCESS_INFORMATION g_pi_go2rtc = {0};
PROCESS_INFORMATION g_pi_ocr = {0};
HANDLE g_sidecar_job = nullptr;
HANDLE g_single_instance_mutex = nullptr;
std::mutex g_sidecar_mutex;

bool sidecarRunning(const PROCESS_INFORMATION& process) {
    if (!process.hProcess) return false;
    DWORD exit_code{};
    return GetExitCodeProcess(process.hProcess, &exit_code) && exit_code == STILL_ACTIVE;
}

void closeSidecarHandles(PROCESS_INFORMATION& process) {
    if (process.hProcess) CloseHandle(process.hProcess);
    if (process.hThread) CloseHandle(process.hThread);
    ZeroMemory(&process, sizeof(process));
}

void ensureSidecarJob() {
    if (g_sidecar_job) return;
    g_sidecar_job = CreateJobObjectA(nullptr, nullptr);
    if (!g_sidecar_job) return;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(g_sidecar_job, JobObjectExtendedLimitInformation,
                                 &limits, sizeof(limits))) {
        CloseHandle(g_sidecar_job);
        g_sidecar_job = nullptr;
    }
}

void stopHiddenProcess(PROCESS_INFORMATION& process) {
    std::lock_guard<std::mutex> lock(g_sidecar_mutex);
    if (sidecarRunning(process)) {
        TerminateProcess(process.hProcess, 0);
        WaitForSingleObject(process.hProcess, 2000);
    }
    closeSidecarHandles(process);
}

void startHiddenProcess(const std::string& command, PROCESS_INFORMATION& process) {
    std::lock_guard<std::mutex> lock(g_sidecar_mutex);
    if (sidecarRunning(process)) return;
    closeSidecarHandles(process);
    ensureSidecarJob();

    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESHOWWINDOW;
    startup.wShowWindow = SW_HIDE;
    std::vector<char> command_line(command.begin(), command.end());
    command_line.push_back('\0');
    PROCESS_INFORMATION candidate{};
    if (!CreateProcessA(nullptr, command_line.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr,
                        &startup, &candidate)) {
        std::cerr << "[SYSTEM ERROR] Sidecar start failed: " << command
                  << " (" << GetLastError() << ")\n";
        return;
    }
    if (g_sidecar_job && !AssignProcessToJobObject(g_sidecar_job, candidate.hProcess)) {
        TerminateProcess(candidate.hProcess, 0);
        closeSidecarHandles(candidate);
        return;
    }
    ResumeThread(candidate.hThread);
    process = candidate;
    std::cout << "[SYSTEM] Sidecar started: " << command
              << " (PID " << process.dwProcessId << ")\n";
}

void restartHiddenProcess(const std::string& command, PROCESS_INFORMATION& process) {
    stopHiddenProcess(process);
    startHiddenProcess(command, process);
}

void sidecarWatchdogLoop() {
    const auto ocr_command = "python " + getPythonScriptPath("supervision_ai_service.py");
    while (g_running.load()) {
        startHiddenProcess("go2rtc.exe", g_pi_go2rtc);
        startHiddenProcess(ocr_command, g_pi_ocr);
        for (int tick = 0; tick < 20 && g_running.load(); ++tick) {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
    }
}

BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_CLOSE_EVENT || signal == CTRL_BREAK_EVENT) {
        g_running.store(false);
        return TRUE;
    }
    return FALSE;
}
#endif

// â”€â”€â”€ MAIN FUNCTION & API CONFIGURATION â”€â”€â”€



bool ensureRuntimeCamera(const std::string& camera_id,
                         const std::string& name,
                         const std::string& rtsp_url) {
    {
        std::lock_guard<std::mutex> lock(g_cameras_mutex);
        const auto existing = std::find_if(
            g_active_cameras.begin(), g_active_cameras.end(),
            [&camera_id](const ActiveCameraInfo& camera) {
                return camera.camera_id == camera_id;
            });
        if (existing != g_active_cameras.end()) return true;
    }

    registerStreamToGo2RTC(camera_id, rtsp_url);
    const auto local_rtsp = "rtsp://127.0.0.1:8554/" + camera_id;
    auto capture = std::make_unique<RTSPCapture>(camera_id, local_rtsp);
    if (!capture->start()) return false;

    std::lock_guard<std::mutex> lock(g_cameras_mutex);
    const auto inserted_by_other_thread = std::find_if(
        g_active_cameras.begin(), g_active_cameras.end(),
        [&camera_id](const ActiveCameraInfo& camera) {
            return camera.camera_id == camera_id;
        });
    if (inserted_by_other_thread == g_active_cameras.end()) {
        g_active_cameras.push_back({camera_id, name, std::move(capture), camera_id});
        g_runtime_camera_ids.insert(camera_id);
    }
    return true;
}

bool stopRuntimeCamera(const std::string& camera_id) {
    bool ai_owned = false;
    {
        std::lock_guard<std::mutex> lock(g_ai_list_mutex);
        ai_owned = g_ai_camera_list.find(camera_id) != g_ai_camera_list.end();
    }
    std::unique_ptr<RTSPCapture> capture;
    {
        std::lock_guard<std::mutex> lock(g_cameras_mutex);
        if (g_runtime_camera_ids.erase(camera_id) == 0) return false;
        if (ai_owned) return true;
        const auto existing = std::find_if(
            g_active_cameras.begin(), g_active_cameras.end(),
            [&camera_id](const ActiveCameraInfo& camera) {
                return camera.camera_id == camera_id;
            });
        if (existing != g_active_cameras.end()) {
            capture = std::move(existing->capture);
            g_active_cameras.erase(existing);
        }
    }
    retireCapture(camera_id, std::move(capture), true);
    return true;
}

int main() {
#ifdef _WIN32
    g_single_instance_mutex = CreateMutexA(nullptr, TRUE, "Local\\VMS_Server_Core_8080");
    if (!g_single_instance_mutex || GetLastError() == ERROR_ALREADY_EXISTS) {
        if (g_single_instance_mutex) CloseHandle(g_single_instance_mutex);
        g_single_instance_mutex = nullptr;
        std::cerr << "[VMS START ERROR] Another instance is already running.\n";
        return 2;
    }
#endif
    // 1. Khá»Ÿi táº¡o CSDL sá»›m Ä‘á»ƒ láº¥y cáº¥u hĂ¬nh camera ban Ä‘áº§u
    if (!initDatabase()) {
        std::cerr << "[VMS START ERROR] Khong the ket noi CSDL PostgreSQL. Vui long kiem tra lai." << std::endl;
        return 1;
    }

    // Ghi cáº¥u hĂ¬nh go2rtc.yaml ban Ä‘áº§u Ä‘á»ƒ go2rtc dĂ¹ng ngay khi khá»Ÿi cháº¡y
    std::vector<std::pair<std::string, std::string>> initial_cams;
    try {
        pqxx::connection conn(DB_CONN_STR);
        pqxx::nontransaction txn(conn);
        pqxx::result res = txn.exec("SELECT camera_id, rtsp_url FROM cameras WHERE is_active = true");
        for (auto row : res) {
            initial_cams.push_back({row[0].as<std::string>(), row[1].as<std::string>()});
        }
    } catch (...) {}
    generateGo2RTCConfig(initial_cams);

#ifdef _WIN32
    SetConsoleCtrlHandler(consoleHandler, TRUE);
    std::cout << "[SYSTEM] Dang khoi dong cac dich vu ngam..." << std::endl;
    startHiddenProcess("go2rtc.exe", g_pi_go2rtc);
    startHiddenProcess("python " + getPythonScriptPath("supervision_ai_service.py"), g_pi_ocr);
    std::thread sidecar_watchdog_thread(sidecarWatchdogLoop);
    std::this_thread::sleep_for(std::chrono::seconds(2));
#endif

    // Khá»Ÿi cháº¡y tiáº¿n trĂ¬nh quĂ©t tráº¡ng thĂ¡i camera
    startStatusPoller();

    // Khá»Ÿi cháº¡y luá»“ng cáº­p nháº­t thĂ´ng sá»‘ tĂ i nguyĂªn ngáº§m
    std::thread resource_thread([]() {
        while (g_running) {
            SystemResourceStats stats = getSystemResourceStats();
            {
                std::lock_guard<std::mutex> lock(g_resources_mutex);
                g_cached_resources = stats;
            }
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    });
    resource_thread.detach();

    // 2. Náº¡p bá»™ nhá»› Ä‘á»‡m khuĂ´n máº·t
    loadFaceIdentitiesCache();

    // 3. Khá»Ÿi táº¡o NVR Engine lÆ°u trá»¯
    std::cout << "[NVR] Khoi tao he thong luu tru..." << std::endl;
    g_nvr_engine = std::make_unique<NVRStorageEngine>("recordings", 50.0);
    std::thread capture_cleanup_thread(captureCleanupLoop);

    // 4. Kh??xi t???o AI Runtime
    try {
        const bool prefer_cuda = envFlagEnabled("VMS_AI_USE_CUDA", true);
        const auto load_model = [prefer_cuda](auto& engine, const std::string& path) {
            if (prefer_cuda && engine.loadModel(path, true)) {
                std::cout << "[AI ENGINE] CUDA provider active for " << path << '\n';
                return true;
            }
            if (prefer_cuda) {
                std::cerr << "[AI ENGINE] CUDA unavailable for " << path
                          << "; falling back to CPU.\n";
            }
            return engine.loadModel(path, false);
        };

        g_vehicle_detector = std::make_unique<ONNXDetector>();
        g_plate_detector = std::make_unique<ONNXDetector>();
        g_face_detector = std::make_unique<ONNXDetector>();
        g_face_embedder = std::make_unique<FaceEmbedder>();
        const bool models_ready =
            load_model(*g_vehicle_detector, "models/yolo11n.onnx") &&
            load_model(*g_plate_detector, "models/best_plateSegment.onnx") &&
            load_model(*g_face_detector, "models/yolo11n-face.onnx") &&
            load_model(*g_face_embedder, "models/face_recognition_sface_2021dec.onnx");
        if (!models_ready) throw std::runtime_error{"One or more AI models failed to load"};
    } catch (const std::exception& e) {
        std::cerr << "[AI INIT ERROR] " << e.what() << std::endl;
        g_ai_enabled.store(false);
    }

    // 5. Náº¡p danh sĂ¡ch Camera tá»« CSDL
    {
        std::lock_guard<std::mutex> lock(g_cameras_mutex);
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::nontransaction txn(conn);
            pqxx::result res = txn.exec("SELECT camera_id, name, rtsp_url, ai_enabled FROM cameras WHERE is_active = true AND ai_enabled = true");
            
            for (auto row : res) {
                std::string cam_id = row[0].as<std::string>();
                std::string name = row[1].as<std::string>();
                std::string rtsp_url = row[2].as<std::string>();
                bool ai_en = row[3].as<bool>();

                // LuĂ´n Ä‘Äƒng kĂ½ luá»“ng vá»›i go2rtc lĂºc khá»Ÿi Ä‘á»™ng
                registerStreamToGo2RTC(cam_id, rtsp_url);

                // Khá»Ÿi cháº¡y camera thĂ´ng qua go2rtc thay vĂ¬ gÂi trá»±c tiáº¿p NVR
                std::string local_rtsp = "rtsp://127.0.0.1:8554/" + cam_id;
                auto capture = std::make_unique<RTSPCapture>(cam_id, local_rtsp);
                if (capture->start()) {
                    g_active_cameras.push_back({cam_id, name, std::move(capture), cam_id});
                    if (ai_en) {
                        std::lock_guard<std::mutex> ai_lock(g_ai_list_mutex);
                        g_ai_camera_list.insert(cam_id);
                    }
                }
            }
            std::cout << "[VMS] Da nap " << g_active_cameras.size() << " cameras tu CSDL." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[DB ERROR] Khong the nap camera: " << e.what() << std::endl;
        }
    }

    std::thread ai_thread(aiWorkerLoop);
    std::thread frame_publisher_thread;
    if (envFlagEnabled("VMS_ENABLE_FRAME_PUBLISHER", false)) {
        frame_publisher_thread = std::thread(framePublisherLoop);
    } else {
        std::cout << "[FRAME SERVICE] ZMQ JPEG publisher disabled (set VMS_ENABLE_FRAME_PUBLISHER=1 to enable).\n";
    }
    std::thread recording_thread([]() {
        const bool recording_enabled = envFlagEnabled("VMS_ENABLE_NVR", true);
        const auto ffmpeg_executable = findFFmpegExecutable();
        auto next_storage_check = std::chrono::steady_clock::now();
        while (g_running.load()) {
            std::vector<std::pair<std::string, std::string>> streams;
            if (recording_enabled) {
                std::lock_guard<std::mutex> lock(g_cameras_mutex);
                streams.reserve(g_active_cameras.size());
                for (const auto& camera : g_active_cameras) {
                    if (!camera.capture || !camera.capture->isOpened()) continue;
                    const auto& stream_id =
                        camera.stream_id.empty() ? camera.camera_id : camera.stream_id;
                    streams.emplace_back(
                        camera.camera_id,
                        "rtsp://127.0.0.1:8554/" + stream_id);
                }
            }
            if (g_nvr_engine) {
                g_nvr_engine->reconcileStreamRecorders(streams, ffmpeg_executable);
                const auto now = std::chrono::steady_clock::now();
                if (now >= next_storage_check) {
                    g_nvr_engine->enforceStorageLimits();
                    next_storage_check = now + std::chrono::minutes{1};
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds{2});
        }
        if (g_nvr_engine) g_nvr_engine->stopAllStreamRecorders();
    });

    // 6. Cáº¥u hĂ¬nh Crow HTTP API & WebSockets Server
    crow::App<CORSMiddleware> app;

    // WebSocket Endpoint cho cáº£nh bĂ¡o sá»± kiá»‡n trá»±c tiáº¿p
    CROW_ROUTE(app, "/api/alerts")
        .websocket(&app)
        .onopen([](crow::websocket::connection& conn) {
            std::lock_guard<std::mutex> lock(g_ws_mutex);
            g_ws_connections.insert(&conn);
            std::cout << "[WS] Client ket noi. Tong so client: " << g_ws_connections.size() << std::endl;
        })
        .onclose([](crow::websocket::connection& conn, const std::string& reason, uint16_t status) {
            std::lock_guard<std::mutex> lock(g_ws_mutex);
            g_ws_connections.erase(&conn);
            std::cout << "[WS] Client ngat ket noi. Tong so client: " << g_ws_connections.size() << " | Status: " << status << std::endl;
        });

    // Endpoint XĂ³a toĂ n bá»™ dá»¯ liá»‡u (Factory Reset)
    CROW_ROUTE(app, "/api/system/factory_reset").methods("POST"_method)([]() {
        try {
            // 1. Truncate database
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            txn.exec("TRUNCATE TABLE events, detections, identities, cameras, dvrs CASCADE;");
            txn.commit();
            
            // 2. Clear in-memory caches safely, stopping active capture threads
            std::vector<std::unique_ptr<RTSPCapture>> captures_to_stop;
            {
                std::lock_guard<std::mutex> lock(g_cameras_mutex);
                for (auto& cam_info : g_active_cameras) {
                    if (cam_info.capture) {
                        captures_to_stop.push_back(std::move(cam_info.capture));
                    }
                }
                g_active_cameras.clear();
                g_runtime_camera_ids.clear();
            }
            for (auto& cap : captures_to_stop) {
                try {
                    cap->stop();
                } catch (...) {}
            }

            {
                std::lock_guard<std::mutex> lock(g_identities_mutex);
                g_face_identities_cache.clear();
            }
            {
                std::lock_guard<std::mutex> lock(g_detections_mutex);
                g_detections_cache.clear();
            }
            {
                std::lock_guard<std::mutex> lock(g_ai_list_mutex);
                g_ai_camera_list.clear();
            }
            
            // 3. Overwrite go2rtc config with empty list and restart go2rtc
            generateGo2RTCConfig({});
#ifdef _WIN32
            restartHiddenProcess("go2rtc.exe", g_pi_go2rtc);
#endif

            // 4. Clear snapshot folder
            try {
                for (const auto& entry : std::filesystem::directory_iterator("snapshots")) {
                    std::filesystem::remove_all(entry.path());
                }
            } catch (...) {}
            
            crow::json::wvalue res;
            res["status"] = "success";
            return crow::response(200, res);
        } catch (const std::exception& e) {
            crow::json::wvalue res;
            res["status"] = "error";
            res["message"] = e.what();
            return crow::response(500, res);
        }
    });

    // Endpoints CRUD Camera

    // Runtime bridge for the PySide desktop app. The C++ service owns RTSP;
    // Python only subscribes to the already-decoded display frames over ZMQ.
    CROW_ROUTE(app, "/api/runtime/cameras/start").methods("POST"_method)([](const crow::request& req) {
        const auto body = crow::json::load(req.body);
        if (!body || !body.has("camera_id") || !body.has("rtsp_url")) {
            return crow::response(400, "Missing camera_id or rtsp_url");
        }

        const std::string camera_id = body["camera_id"].s();
        const std::string name = body.has("name") ? std::string(body["name"].s()) : camera_id;
        const std::string rtsp_url = body["rtsp_url"].s();
        if (camera_id.empty() || rtsp_url.empty()) {
            return crow::response(400, "camera_id and rtsp_url must not be empty");
        }

        if (!ensureRuntimeCamera(camera_id, name, rtsp_url)) {
            return crow::response(500, "Unable to start C++ camera capture");
        }

        crow::json::wvalue result;
        result["status"] = "running";
        result["camera_id"] = camera_id;
        result["frame_endpoint"] = kFramePublisherEndpoint;
        return crow::response(200, result);
    });

    CROW_ROUTE(app, "/api/runtime/cameras/stop").methods("POST"_method)([](const crow::request& req) {
        const auto body = crow::json::load(req.body);
        if (!body || !body.has("camera_id")) {
            return crow::response(400, "Missing camera_id");
        }
        const std::string camera_id = body["camera_id"].s();
        crow::json::wvalue result;
        result["camera_id"] = camera_id;
        result["stopped"] = stopRuntimeCamera(camera_id);
        return crow::response(200, result);
    });

    CROW_ROUTE(app, "/api/cameras")([]() {
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::nontransaction txn(conn);
            pqxx::result r = txn.exec("SELECT camera_id, name, rtsp_url, is_active, grid_position, dvr_id, ip_address, port, username, password, brand, is_ip_camera FROM cameras");
            std::vector<crow::json::wvalue> cams_json;
            for (auto row : r) {
                crow::json::wvalue cam;
                cam["camera_id"] = row["camera_id"].as<std::string>();
                cam["name"] = row["name"].as<std::string>();
                cam["rtsp_url"] = row["rtsp_url"].as<std::string>();
                cam["is_active"] = row["is_active"].as<bool>();
                cam["grid_position"] = row["grid_position"].as<int>();
                cam["dvr_id"] = row["dvr_id"].is_null() ? "" : row["dvr_id"].as<std::string>();
                cam["ip_address"] = row["ip_address"].is_null() ? "" : row["ip_address"].as<std::string>();
                cam["port"] = row["port"].is_null() ? 0 : row["port"].as<int>();
                cam["username"] = row["username"].is_null() ? "" : row["username"].as<std::string>();
                cam["password"] = row["password"].is_null() ? "" : row["password"].as<std::string>();
                cam["brand"] = row["brand"].is_null() ? "" : row["brand"].as<std::string>();
                cam["is_ip_camera"] = row["is_ip_camera"].is_null() ? false : row["is_ip_camera"].as<bool>();
                cams_json.push_back(std::move(cam));
            }
            return crow::response(crow::json::wvalue(cams_json));
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint láº¥y tráº¡ng thĂ¡i online/offline thá»±c táº¿ cá»§a camera tá»« Ä‘áº§u ghi NVR
    CROW_ROUTE(app, "/api/cameras/status")([]() {
        try {
            std::ifstream file("web/camera_status.json");
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                file.close();
                auto parsed = crow::json::load(buffer.str());
                if (parsed) {
                    return crow::response(200, "application/json", buffer.str());
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[STATUS ERROR] " << e.what() << std::endl;
        }

        // Fallback to offline/online based on active C++ captures if file isn't available or fails to parse
        crow::json::wvalue fallback = crow::json::wvalue::object();
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::nontransaction txn(conn);
            pqxx::result r = txn.exec("SELECT camera_id FROM cameras");
            
            std::unordered_set<std::string> online_cams;
            {
                std::lock_guard<std::mutex> lock(g_cameras_mutex);
                for (const auto& cam : g_active_cameras) {
                    if (cam.capture && cam.capture->isOpened()) {
                        online_cams.insert(cam.camera_id);
                    }
                }
            }

            for (auto row : r) {
                std::string cam_id = row["camera_id"].as<std::string>();
                if (online_cams.count(cam_id)) {
                    fallback[cam_id] = "online";
                } else {
                    fallback[cam_id] = "offline";
                }
            }
        } catch (...) {}
        return crow::response(fallback);
    });

    CROW_ROUTE(app, "/api/cameras").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Bad Request: Invalid JSON");
        std::string cam_id = body["camera_id"].s();
        std::string name = body["name"].s();
        std::string rtsp_url = body.has("rtsp_url") && body["rtsp_url"].t() == crow::json::type::String ? std::string(body["rtsp_url"].s()) : "";
        
        bool is_ip_camera = body.has("is_ip_camera") ? body["is_ip_camera"].b() : false;
        std::optional<std::string> ip_address = std::nullopt;
        if (body.has("ip_address") && body["ip_address"].t() == crow::json::type::String) ip_address = std::string(body["ip_address"].s());
        
        std::optional<int> port = std::nullopt;
        if (body.has("port") && body["port"].t() == crow::json::type::Number) port = static_cast<int>(body["port"].i());
        
        std::optional<std::string> username = std::nullopt;
        if (body.has("username") && body["username"].t() == crow::json::type::String) username = std::string(body["username"].s());
        
        std::optional<std::string> password = std::nullopt;
        if (body.has("password") && body["password"].t() == crow::json::type::String) password = std::string(body["password"].s());
        
        std::optional<std::string> brand = std::nullopt;
        if (body.has("brand") && body["brand"].t() == crow::json::type::String) brand = std::string(body["brand"].s());

        if (is_ip_camera && ip_address && port && username && password && brand) {
            std::string usr = urlEncode(*username);
            std::string pwd = urlEncode(*password);
            if (*brand == "hikvision") {
                rtsp_url = "rtsp://" + usr + ":" + pwd + "@" + *ip_address + ":" + std::to_string(normalizeRtspPort(*port)) + "/Streaming/Channels/101";
            } else if (*brand == "dahua" || *brand == "kbvision") {
                rtsp_url = "rtsp://" + usr + ":" + pwd + "@" + *ip_address + ":" + std::to_string(normalizeRtspPort(*port)) + "/cam/realmonitor?channel=1&subtype=0";
            } else if (*brand == "dvrip") {
                rtsp_url = "dvrip://" + usr + ":" + pwd + "@" + *ip_address + ":" + std::to_string(*port) + "?channel=1&subtype=1";
            } else {
                rtsp_url = "rtsp://" + usr + ":" + pwd + "@" + *ip_address + ":" + std::to_string(normalizeRtspPort(*port)) + "/onvif1";
            }
        }

        std::optional<std::string> dvr_opt;
        if (body.has("dvr_id")) {
            std::string dvr_id = body["dvr_id"].s();
            if (!dvr_id.empty() && dvr_id != "null") {
                dvr_opt = dvr_id;
            }
        }
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("add_camera", "INSERT INTO cameras (camera_id, name, rtsp_url, dvr_id, ip_address, port, username, password, brand, is_ip_camera) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)");
            txn.exec_prepared("add_camera", cam_id, name, rtsp_url, dvr_opt, ip_address, port, username, password, brand, is_ip_camera);
            txn.commit();

            // 1. G i API cá»§a go2rtc Ä‘á»ƒ náº¡p luá»“ng camera má»›i Ä‘á»™ng (SD vĂ  HD) á»Ÿ runtime trÆ°á»›c tiĂªn
            registerStreamToGo2RTC(cam_id, rtsp_url);
            std::cout << "[VMS SIDE] Dang ky luong dong (SD va HD) len go2rtc" << std::endl;

            return crow::response(201, "Camera added successfully");
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoints quáº£n lĂ½ Ä‘áº§u ghi hĂ¬nh NVR/DVR
    CROW_ROUTE(app, "/api/dvrs")([]() {
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::nontransaction txn(conn);
            pqxx::result r = txn.exec("SELECT dvr_id, name, host, http_port, rtsp_port, tcp_port, username, brand FROM dvrs");
            std::vector<crow::json::wvalue> dvrs_json;
            for (auto row : r) {
                crow::json::wvalue dvr;
                dvr["dvr_id"] = row["dvr_id"].as<std::string>();
                dvr["name"] = row["name"].as<std::string>();
                dvr["host"] = row["host"].as<std::string>();
                dvr["http_port"] = row["http_port"].as<int>();
                dvr["rtsp_port"] = row["rtsp_port"].as<int>();
                dvr["tcp_port"] = row["tcp_port"].as<int>();
                dvr["username"] = row["username"].as<std::string>();
                dvr["brand"] = row["brand"].as<std::string>();
                dvrs_json.push_back(std::move(dvr));
            }
            return crow::response(crow::json::wvalue(dvrs_json));
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    CROW_ROUTE(app, "/api/dvrs").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Bad Request: Invalid JSON");
        std::string dvr_id = body["dvr_id"].s();
        std::string name = body["name"].s();
        std::string host = body["host"].s();
        int http_port = body.has("http_port") ? body["http_port"].i() : 80;
        int rtsp_port = body.has("rtsp_port") ? body["rtsp_port"].i() : 554;
        int tcp_port = body.has("tcp_port") ? body["tcp_port"].i() : 37777;
        std::string username = body.has("username") ? std::string(body["username"].s()) : "admin";
        std::string password = body["password"].s();
        std::string brand = body.has("brand") ? std::string(body["brand"].s()) : "generic";
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("add_dvr", "INSERT INTO dvrs (dvr_id, name, host, http_port, rtsp_port, tcp_port, username, password, brand) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)");
            txn.exec_prepared("add_dvr", dvr_id, name, host, http_port, rtsp_port, tcp_port, username, password, brand);
            txn.commit();
            // Tá»± Ä‘á»™ng Ä‘á»“ng bá»™ cĂ¡c kĂªnh camera sau khi thĂªm thĂ nh cĂ´ng
            std::thread([dvr_id, host, http_port, rtsp_port, username, password, brand]() {
                syncDVRCameras(dvr_id, host, http_port, rtsp_port, username, password, brand);
            }).detach();
            return crow::response(201, "DVR added successfully");
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    CROW_ROUTE(app, "/api/dvrs/sync").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body || !body.has("dvr_id")) return crow::response(400, "Bad Request: Missing dvr_id");
        std::string dvr_id = body["dvr_id"].s();
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("get_dvr", "SELECT host, http_port, rtsp_port, username, password, brand FROM dvrs WHERE dvr_id = $1");
            pqxx::result r = txn.exec_prepared("get_dvr", dvr_id);
            if (r.empty()) {
                return crow::response(404, "DVR not found");
            }
            auto row = r[0];
            std::string host = row["host"].as<std::string>();
            int http_port = row["http_port"].as<int>();
            int rtsp_port = row["rtsp_port"].as<int>();
            std::string username = row["username"].as<std::string>();
            std::string password = row["password"].as<std::string>();
            std::string brand = row["brand"].as<std::string>();
            txn.commit();
            int count = syncDVRCameras(dvr_id, host, http_port, rtsp_port, username, password, brand);
            crow::json::wvalue res;
            res["status"] = "success";
            res["added_count"] = count;
            return crow::response(200, res);
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint Ä‘Äƒng kĂ½ biá»ƒn sá»‘ / khuĂ´n máº·t máº«u
    
    // Endpoint Cáº­p nháº­t Camera (PUT)
    CROW_ROUTE(app, "/api/cameras").methods("PUT"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Bad Request: Invalid JSON");
        std::string cam_id = body["camera_id"].s();
        std::string name = body["name"].s();
        std::string rtsp_url = body.has("rtsp_url") && body["rtsp_url"].t() == crow::json::type::String ? std::string(body["rtsp_url"].s()) : "";
        
        bool is_ip_camera = body.has("is_ip_camera") ? body["is_ip_camera"].b() : false;
        std::optional<std::string> ip_address = std::nullopt;
        if (body.has("ip_address") && body["ip_address"].t() == crow::json::type::String) ip_address = std::string(body["ip_address"].s());
        
        std::optional<int> port = std::nullopt;
        if (body.has("port") && body["port"].t() == crow::json::type::Number) port = static_cast<int>(body["port"].i());
        
        std::optional<std::string> username = std::nullopt;
        if (body.has("username") && body["username"].t() == crow::json::type::String) username = std::string(body["username"].s());
        
        std::optional<std::string> password = std::nullopt;
        if (body.has("password") && body["password"].t() == crow::json::type::String) password = std::string(body["password"].s());
        
        std::optional<std::string> brand = std::nullopt;
        if (body.has("brand") && body["brand"].t() == crow::json::type::String) brand = std::string(body["brand"].s());

        if (is_ip_camera && ip_address && port && username && password && brand) {
            std::string usr = urlEncode(*username);
            std::string pwd = urlEncode(*password);
            if (*brand == "hikvision") {
                rtsp_url = "rtsp://" + usr + ":" + pwd + "@" + *ip_address + ":" + std::to_string(normalizeRtspPort(*port)) + "/Streaming/Channels/101";
            } else if (*brand == "dahua" || *brand == "kbvision") {
                rtsp_url = "rtsp://" + usr + ":" + pwd + "@" + *ip_address + ":" + std::to_string(normalizeRtspPort(*port)) + "/cam/realmonitor?channel=1&subtype=0";
            } else if (*brand == "dvrip") {
                rtsp_url = "dvrip://" + usr + ":" + pwd + "@" + *ip_address + ":" + std::to_string(*port) + "?channel=1&subtype=1";
            } else {
                rtsp_url = "rtsp://" + usr + ":" + pwd + "@" + *ip_address + ":" + std::to_string(normalizeRtspPort(*port)) + "/onvif1";
            }
        }

        std::optional<std::string> dvr_opt;
        if (body.has("dvr_id")) {
            std::string dvr_id = body["dvr_id"].s();
            if (!dvr_id.empty() && dvr_id != "null") {
                dvr_opt = dvr_id;
            }
        }
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("update_camera", "UPDATE cameras SET name=$1, rtsp_url=$2, dvr_id=$3, ip_address=$4, port=$5, username=$6, password=$7, brand=$8, is_ip_camera=$9 WHERE camera_id=$10");
            pqxx::result r = txn.exec_prepared("update_camera", name, rtsp_url, dvr_opt, ip_address, port, username, password, brand, is_ip_camera, cam_id);
            if (r.affected_rows() == 0) {
                return crow::response(404, "Camera not found");
            }
            txn.commit();

            // XĂ³a luá»“ng cÅ© trĂªn go2rtc (cáº£ SD vĂ  HD)
#ifdef _WIN32
            std::string curl_cmd_del1 = "curl.exe -s -X DELETE \"http://localhost:1984/api/streams?src=" + cam_id + "\"";
            std::string curl_cmd_del2 = "curl.exe -s -X DELETE \"http://localhost:1984/api/streams?src=" + cam_id + "_hd\"";
#else
            std::string curl_cmd_del1 = "curl -s -X DELETE \"http://localhost:1984/api/streams?src=" + cam_id + "\"";
            std::string curl_cmd_del2 = "curl -s -X DELETE \"http://localhost:1984/api/streams?src=" + cam_id + "_hd\"";
#endif
            system(("\"" + curl_cmd_del1 + "\"").c_str());
            system(("\"" + curl_cmd_del2 + "\"").c_str());

            // ÂÄƒng kĂ½ luá»“ng má»›i (SD vĂ  HD)
            registerStreamToGo2RTC(cam_id, rtsp_url);

            // Khá»Ÿi á»™ng láº¡i luá»“ng trong C++ náº¿u camera ang hoáº¡t á»™ng (Stop old, start new)
            bool was_active = false;
            std::lock_guard<std::mutex> lock(g_cameras_mutex);
            for (auto it = g_active_cameras.begin(); it != g_active_cameras.end(); ++it) {
                if (it->camera_id == cam_id) {
                    if (it->capture) {
                        it->capture->stop();
                    }
                    g_active_cameras.erase(it);
                    was_active = true;
                    break;
                }
            }
            if (was_active) {
                std::string local_rtsp = "rtsp://127.0.0.1:8554/" + cam_id;
                auto capture = std::make_unique<RTSPCapture>(cam_id, local_rtsp);
                if (capture->start()) {
                    g_active_cameras.push_back({cam_id, name, std::move(capture), cam_id});
                }
            }

            return crow::response(200, "Camera updated successfully");
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint XĂ³a Camera (DELETE)
    CROW_ROUTE(app, "/api/cameras/<string>").methods("DELETE"_method)([](const std::string& cam_id) {
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("delete_camera", "DELETE FROM cameras WHERE camera_id=$1");
            pqxx::result r = txn.exec_prepared("delete_camera", cam_id);
            if (r.affected_rows() == 0) {
                return crow::response(404, "Camera not found");
            }
            txn.commit();

            // XĂ³a khÂi go2rtc (cáº£ SD vĂ  HD)
#ifdef _WIN32
            std::string curl_cmd_del1 = "curl.exe -s -X DELETE \"http://localhost:1984/api/streams?src=" + cam_id + "\"";
            std::string curl_cmd_del2 = "curl.exe -s -X DELETE \"http://localhost:1984/api/streams?src=" + cam_id + "_hd\"";
#else
            std::string curl_cmd_del1 = "curl -s -X DELETE \"http://localhost:1984/api/streams?src=" + cam_id + "\"";
            std::string curl_cmd_del2 = "curl -s -X DELETE \"http://localhost:1984/api/streams?src=" + cam_id + "_hd\"";
#endif
            system(("\"" + curl_cmd_del1 + "\"").c_str());
            system(("\"" + curl_cmd_del2 + "\"").c_str());

            // Dá»«ng luá»“ng trong C++
            std::lock_guard<std::mutex> lock(g_cameras_mutex);
            for (auto it = g_active_cameras.begin(); it != g_active_cameras.end(); ++it) {
                if (it->camera_id == cam_id) {
                    if (it->capture) {
                        it->capture->stop();
                    }
                    g_active_cameras.erase(it);
                    break;
                }
            }

            return crow::response(200, "Camera deleted successfully");
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint Cáº­p nháº­t Âáº§u ghi (PUT)
    CROW_ROUTE(app, "/api/dvrs").methods("PUT"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Bad Request: Invalid JSON");
        std::string dvr_id = body["dvr_id"].s();
        std::string name = body["name"].s();
        std::string host = body["host"].s();
        int http_port = body.has("http_port") ? body["http_port"].i() : 80;
        int rtsp_port = body.has("rtsp_port") ? body["rtsp_port"].i() : 554;
        int tcp_port = body.has("tcp_port") ? body["tcp_port"].i() : 37777;
        std::string username = body.has("username") ? std::string(body["username"].s()) : "admin";
        std::string password = body.has("password") && body["password"].t() == crow::json::type::String ? std::string(body["password"].s()) : "";
        std::string brand = body.has("brand") ? std::string(body["brand"].s()) : "generic";
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("update_dvr", "UPDATE dvrs SET name=$1, host=$2, http_port=$3, rtsp_port=$4, tcp_port=$5, username=$6, password=$7, brand=$8 WHERE dvr_id=$9");
            pqxx::result r = txn.exec_prepared("update_dvr", name, host, http_port, rtsp_port, tcp_port, username, password, brand, dvr_id);
            if (r.affected_rows() == 0) {
                return crow::response(404, "DVR not found");
            }
            txn.commit();
            return crow::response(200, "DVR updated successfully");
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint XĂ³a Âáº§u ghi (DELETE)
    CROW_ROUTE(app, "/api/dvrs/<string>").methods("DELETE"_method)([](const std::string& dvr_id) {
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("delete_dvr", "DELETE FROM dvrs WHERE dvr_id=$1");
            pqxx::result r = txn.exec_prepared("delete_dvr", dvr_id);
            if (r.affected_rows() == 0) {
                return crow::response(404, "DVR not found");
            }
            txn.commit();
            return crow::response(200, "DVR deleted successfully");
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });


    CROW_ROUTE(app, "/api/identities").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Bad Request: Invalid JSON");
        std::string type = body["identity_type"].s();
        std::string label = body["label"].s();
        std::string data_path = body["data_path"].s();
        bool is_whitelisted = body["is_whitelisted"].b();
        std::string emb_str = "";
        if (body.has("face_embedding")) {
            auto emb_list = body["face_embedding"];
            std::stringstream ss;
            for (size_t i = 0; i < emb_list.size(); ++i) {
                ss << emb_list[i].d() << (i == emb_list.size() - 1 ? "" : ",");
            }
            emb_str = ss.str();
        }
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            std::optional<std::string> emb_opt;
            if (!emb_str.empty()) {
                emb_opt = emb_str;
            }
            conn.prepare("add_identity", "INSERT INTO identities (identity_type, label, data_path, is_whitelisted, face_embedding) VALUES ($1, $2, $3, $4, $5)");
            txn.exec_prepared("add_identity", type, label, data_path, is_whitelisted, emb_opt);
            txn.commit();
            if (type == "face" && !emb_str.empty()) {
                loadFaceIdentitiesCache();
            }
            return crow::response(201, "Identity registered successfully");
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    
    // Endpoint truy xuat lich su nhan dien chi tiet (cho tab Lich su - detection-history)
    CROW_ROUTE(app, "/api/detection-history").methods("GET"_method)([](const crow::request& req) {
        auto type_param = req.url_params.get("type"); // "vehicle", "face", "plate"
        auto keyword = req.url_params.get("keyword");
        auto start_time = req.url_params.get("start_time");
        auto end_time = req.url_params.get("end_time");
        auto limit_str = req.url_params.get("limit");
        
        int limit = 240;
        if (limit_str) {
            try { limit = std::stoi(limit_str); } catch(...) {}
        }
        
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::nontransaction txn(conn);
            
            std::string sql = "SELECT id, camera_id, EXTRACT(EPOCH FROM detected_at)::double precision as timestamp, object_type, "
                              "plate_text, plate_confidence, identity_label, identity_confidence, "
                              "vehicle_image_path, plate_image_path, face_image_path, location_name "
                              "FROM detections WHERE 1=1";
            
            if (type_param) {
                std::string t(type_param);
                if (t == "vehicle") {
                    sql += " AND object_type LIKE 'vehicle%'";
                } else if (t == "face") {
                    sql += " AND (face_image_path != '' OR identity_label != '')";
                } else if (t == "plate") {
                    sql += " AND (plate_text != '' OR plate_image_path != '')";
                }
            }
            
            if (keyword && std::string(keyword) != "") {
                std::string escaped = "%" + std::string(keyword) + "%";
                sql += " AND (plate_text ILIKE " + txn.quote(escaped) + 
                       " OR identity_label ILIKE " + txn.quote(escaped) + 
                       " OR camera_id ILIKE " + txn.quote(escaped) + 
                       " OR location_name ILIKE " + txn.quote(escaped) + ")";
            }
            if (start_time && std::string(start_time) != "") {
                sql += " AND detected_at >= " + txn.quote(start_time);
            }
            if (end_time && std::string(end_time) != "") {
                sql += " AND detected_at <= " + txn.quote(end_time);
            }
            
            sql += " ORDER BY detected_at DESC LIMIT " + std::to_string(limit);
            
            pqxx::result r = txn.exec(sql);
            std::vector<crow::json::wvalue> list_json;
            for (size_t i = 0; i < r.size(); i++) {
                crow::json::wvalue row;
                row["id"] = r[i][0].as<int>();
                row["camera_id"] = r[i][1].c_str();
                row["timestamp"] = r[i][2].as<double>();
                row["object_type"] = r[i][3].c_str();
                row["plate_text"] = r[i][4].c_str();
                row["plate_confidence"] = r[i][5].as<float>();
                row["identity_label"] = r[i][6].c_str();
                row["identity_confidence"] = r[i][7].as<float>();
                row["vehicle_image_path"] = r[i][8].c_str();
                row["plate_image_path"] = r[i][9].c_str();
                row["face_image_path"] = r[i][10].c_str();
                row["location_name"] = r[i][11].is_null() ? "" : r[i][11].c_str();
                list_json.push_back(std::move(row));
            }
            return crow::response(crow::json::wvalue(list_json));
            
        } catch (const std::exception& e) {
            std::cerr << "[DB ERROR] detection-history: " << e.what() << std::endl;
            return crow::response(500, e.what());
        }
    });


    // Endpoint truy xuáº¥t sá»± kiá»‡n phĂ¡t hiá»‡n AI (Há»— trá»£ lÂc nĂ¢ng cao cho TĂ¬m kiáº¿m lá»‹ch sá»­)
    CROW_ROUTE(app, "/api/events").methods("GET"_method)([](const crow::request& req) {
        auto camera_id = req.url_params.get("camera_id");
        auto event_type = req.url_params.get("event_type");
        auto keyword = req.url_params.get("keyword");
        auto start_time = req.url_params.get("start_time");
        auto end_time = req.url_params.get("end_time");
        auto limit_str = req.url_params.get("limit");
        int limit = 200;
        if (limit_str) {
            try { limit = std::stoi(limit_str); } catch (...) { limit = 200; }
            if (limit < 1) limit = 1;
            if (limit > 500) limit = 500;
        }
        
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::nontransaction txn(conn);
            
            std::string sql = "SELECT id, camera_id, event_type, description, confidence, snapshot_path, plate_text, plate_image_path, EXTRACT(EPOCH FROM timestamp)::double precision as epoch_time FROM events WHERE 1=1";
            
            if (camera_id && std::string(camera_id) != "all" && std::string(camera_id) != "") {
                sql += " AND camera_id = " + txn.quote(camera_id);
            }
            if (event_type && std::string(event_type) != "all" && std::string(event_type) != "") {
                sql += " AND event_type = " + txn.quote(event_type);
            }
            if (keyword && std::string(keyword) != "") {
                std::string escaped = "%" + std::string(keyword) + "%";
                sql += " AND (description ILIKE " + txn.quote(escaped) + " OR camera_id ILIKE " + txn.quote(escaped) + ")";
            }
            if (start_time && std::string(start_time) != "") {
                sql += " AND timestamp >= " + txn.quote(start_time);
            }
            if (end_time && std::string(end_time) != "") {
                sql += " AND timestamp <= " + txn.quote(end_time);
            }
            
            sql += " ORDER BY timestamp DESC LIMIT " + std::to_string(limit);
            
            pqxx::result r = txn.exec(sql);
            
            std::vector<crow::json::wvalue> events_json;
            const auto stored_image_exists = [](const std::string& image_path) {
                if (image_path.empty()) return false;
                const auto filename = std::filesystem::path(image_path).filename();
                if (filename.empty()) return false;
                std::error_code error;
                const auto full_path = std::filesystem::path("snapshots") / "detections" / filename;
                return std::filesystem::is_regular_file(full_path, error);
            };
            for (auto row : r) {
                crow::json::wvalue ev;
                ev["id"] = row["id"].as<int>();
                ev["camera_id"] = row["camera_id"].as<std::string>();
                ev["event_type"] = row["event_type"].as<std::string>();
                ev["description"] = row["description"].as<std::string>();
                ev["confidence"] = row["confidence"].as<float>();
                const std::string snapshot_path = row["snapshot_path"].as<std::string>();
                const std::string plate_image_path = row["plate_image_path"].is_null() ? "" : row["plate_image_path"].as<std::string>();
                ev["snapshot_path"] = snapshot_path;
                ev["plate_text"] = row["plate_text"].is_null() ? "" : row["plate_text"].as<std::string>();
                ev["plate_image_path"] = plate_image_path;
                ev["image_available"] = stored_image_exists(snapshot_path) || stored_image_exists(plate_image_path);
                ev["timestamp"] = row["epoch_time"].as<double>(); // Tráº£ vÂ timestamp dáº¡ng epoch seconds Ä‘á»ƒ JS format dá»… dĂ ng
                events_json.push_back(std::move(ev));
            }
            return crow::response(crow::json::wvalue(events_json));
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint truy xuáº¥t káº¿t quáº£ nháº­n diá»‡n AI (Há»— trá»£ phĂ¢n trang, lÂc nĂ¢ng cao)
    CROW_ROUTE(app, "/api/detections")([]() {
        std::vector<crow::json::wvalue> list_json;
        {
            std::lock_guard<std::mutex> lock(g_detections_mutex);
            for (auto const& [cam_id, det] : g_detections_cache) {
                // Chá»‰ tráº£ vÂ cĂ¡c detect gáº§n Ä‘Ă¢y (< 3 giĂ¢y) Ä‘á»ƒ trĂ¡nh giá»¯ box quĂ¡ lĂ¢u
                if (time(nullptr) - det.timestamp > 3) continue;
                
                crow::json::wvalue cd;
                cd["camera_id"] = cam_id;
                cd["frame_width"] = det.frame_width;
                cd["frame_height"] = det.frame_height;
                
                std::vector<crow::json::wvalue> boxes_json;
                for (auto const& box : det.boxes) {
                    crow::json::wvalue b;
                    b["x"] = box.x;
                    b["y"] = box.y;
                    b["w"] = box.w;
                    b["h"] = box.h;
                    b["label"] = box.label;
                    b["confidence"] = box.confidence;
                    boxes_json.push_back(std::move(b));
                }
                cd["boxes"] = std::move(boxes_json);
                list_json.push_back(std::move(cd));
            }
        }
        return crow::response(crow::json::wvalue(list_json));
    });

    // Endpoint cáº­p nháº­t káº¿t quáº£ nháº­n diá»‡n AI tá»« Python Sidecar (POST)
    CROW_ROUTE(app, "/api/detections").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Bad Request: Invalid JSON");
        try {
            std::string cam_id = body["camera_id"].s();
            int w = body["frame_width"].i();
            int h = body["frame_height"].i();
            std::vector<DetectionBox> boxes;
            if (body.has("boxes")) {
                for (auto& item : body["boxes"]) {
                    DetectionBox box;
                    box.x = item["x"].i();
                    box.y = item["y"].i();
                    box.w = item["w"].i();
                    box.h = item["h"].i();
                    box.label = item["label"].s();
                    box.confidence = static_cast<float>(item["confidence"].d());
                    boxes.push_back(box);
                }
            }
            {
                std::lock_guard<std::mutex> lock(g_detections_mutex);
                g_detections_cache[cam_id] = {cam_id, boxes, time(nullptr), w, h};
            }
            ++g_ai_stats.frames_processed;

            return crow::response(200, "Detections updated successfully");
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint quáº£n lĂ½ danh sĂ¡ch camera Ä‘ang cháº¡y AI on-demand
    CROW_ROUTE(app, "/api/ai/cameras").methods("GET"_method)([]() {
        std::vector<std::string> ai_cameras;
        {
            std::lock_guard<std::mutex> lock(g_ai_list_mutex);
            ai_cameras.assign(g_ai_camera_list.begin(), g_ai_camera_list.end());
        }
        std::vector<crow::json::wvalue> list_json;
        for (const auto& cam_id : ai_cameras) {
            list_json.push_back(cam_id);
        }
        return crow::response(crow::json::wvalue(list_json));
    });

    CROW_ROUTE(app, "/api/ai/modules").methods("GET"_method)([]() {
        crow::json::wvalue modules;
        crow::json::wvalue v_face; v_face["enabled"] = g_ai_face_enabled.load(); modules["face"] = std::move(v_face);
        crow::json::wvalue v_plate; v_plate["enabled"] = g_ai_plate_enabled.load(); modules["plate"] = std::move(v_plate);
        crow::json::wvalue v_ocr; v_ocr["enabled"] = g_ai_plate_enabled.load(); modules["ocr"] = std::move(v_ocr);
        crow::json::wvalue v_vehicle; v_vehicle["enabled"] = g_ai_object_enabled.load(); modules["vehicle"] = std::move(v_vehicle);
        crow::json::wvalue v_person; v_person["enabled"] = g_ai_object_enabled.load(); modules["person"] = std::move(v_person);
        return crow::response(modules);
    });

    CROW_ROUTE(app, "/api/ai/modules").methods("POST"_method)([](const crow::request& req) {
        const auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");

        if (body.has("module") && body.has("enabled")) {
            std::string mod = body["module"].s();
            bool enabled = body["enabled"].b();
            if (mod == "face") g_ai_face_enabled.store(enabled);
            else if (mod == "plate" || mod == "ocr") g_ai_plate_enabled.store(enabled);
            else if (mod == "vehicle" || mod == "person" || mod == "object") g_ai_object_enabled.store(enabled);
            else if (mod == "motion") g_ai_motion_enabled.store(enabled);
        } else {
            if (body.has("face")) g_ai_face_enabled.store(body["face"].b());
            if (body.has("plate")) g_ai_plate_enabled.store(body["plate"].b());
            if (body.has("object")) g_ai_object_enabled.store(body["object"].b());
            if (body.has("motion")) g_ai_motion_enabled.store(body["motion"].b());
        }

        crow::json::wvalue modules;
        crow::json::wvalue v_face; v_face["enabled"] = g_ai_face_enabled.load(); modules["face"] = std::move(v_face);
        crow::json::wvalue v_plate; v_plate["enabled"] = g_ai_plate_enabled.load(); modules["plate"] = std::move(v_plate);
        crow::json::wvalue v_ocr; v_ocr["enabled"] = g_ai_plate_enabled.load(); modules["ocr"] = std::move(v_ocr);
        crow::json::wvalue v_vehicle; v_vehicle["enabled"] = g_ai_object_enabled.load(); modules["vehicle"] = std::move(v_vehicle);
        crow::json::wvalue v_person; v_person["enabled"] = g_ai_object_enabled.load(); modules["person"] = std::move(v_person);
        
        crow::json::wvalue res;
        res["modules"] = std::move(modules);
        return crow::response(res);
    });

    CROW_ROUTE(app, "/api/ai/control").methods("GET"_method)([]() {
        crow::json::wvalue state;
        size_t ai_count = 0;
        {
            std::lock_guard<std::mutex> lock(g_ai_list_mutex);
            ai_count = g_ai_camera_list.size();
        }
        const auto runtime = getAiRuntimeSummary();
        state["enabled"] = g_ai_enabled.load();
        state["engine"] = "cpp-onnxruntime";
        state["ai_cameras"] = static_cast<int>(ai_count);
        state["active_ai_cameras"] = runtime.active_cameras;
        state["live_boxes"] = runtime.live_boxes;
        state["last_frame_age_seconds"] = runtime.last_frame_age_seconds;
        state["frames_processed"] = g_ai_stats.frames_processed.load();
        state["motion_enabled"] = g_ai_motion_enabled.load();
        state["motion_frames_checked"] = g_ai_stats.motion_frames_checked.load();
        state["motion_frames_triggered"] = g_ai_stats.motion_frames_triggered.load();
        state["motion_frames_skipped"] = g_ai_stats.motion_frames_skipped.load();
        state["motion_frames_queued"] = g_ai_stats.motion_frames_queued.load();
        state["plate_attempts"] = g_ai_stats.plate_attempts.load();
        state["plate_boxes_detected"] =
            g_ai_stats.plate_boxes_detected.load();
        state["plate_retry_successes"] =
            g_ai_stats.plate_retry_successes.load();
        return crow::response(state);
    });

    CROW_ROUTE(app, "/api/ai/control").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body || !body.has("enabled")) {
            return crow::response(400, "Bad Request: Missing enabled");
        }
        bool enabled = body["enabled"].b();
        g_ai_enabled = enabled;
        if (!enabled) {
            g_ai_batch_generation.fetch_add(1);
            std::lock_guard<std::mutex> lock(g_detections_mutex);
            g_detections_cache.clear();
        }
        crow::json::wvalue res;
        res["enabled"] = g_ai_enabled.load();
        return crow::response(200, res);
    });

    CROW_ROUTE(app, "/api/ai/cameras").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Bad Request: Invalid JSON");
        if (!body.has("camera_id") || !body.has("enabled")) {
            return crow::response(400, "Bad Request: Missing camera_id or enabled");
        }
        std::string cam_id = body["camera_id"].s();
        std::string stream_id = body.has("stream_id") ? std::string(body["stream_id"].s()) : cam_id;
        if (stream_id.empty()) stream_id = cam_id;
        bool enabled = body["enabled"].b();

        setAICameraState(cam_id, "", stream_id, enabled);
        return crow::response(200, "AI camera state updated successfully");
    });

    CROW_ROUTE(app, "/api/ai/cameras/all").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        bool enabled = !body || !body.has("enabled") ? true : body["enabled"].b();
        std::string quality = body && body.has("quality") ? std::string(body["quality"].s()) : "sd";
        const auto batch_generation = g_ai_batch_generation.fetch_add(1) + 1;
        int changed = 0;

        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::nontransaction txn(conn);
            pqxx::result r = txn.exec("SELECT camera_id, name FROM cameras WHERE is_active=true ORDER BY camera_id");
            if (!enabled) {
                {
                    std::lock_guard<std::mutex> lock(g_ai_list_mutex);
                    changed = static_cast<int>(g_ai_camera_list.size());
                    g_ai_camera_list.clear();
                }
                std::vector<std::pair<std::string, std::unique_ptr<RTSPCapture>>> retired;
                {
                    std::lock_guard<std::mutex> lock(g_cameras_mutex);
                    auto camera = g_active_cameras.begin();
                    while (camera != g_active_cameras.end()) {
                        if (g_runtime_camera_ids.find(camera->camera_id) !=
                            g_runtime_camera_ids.end()) {
                            ++camera;
                            continue;
                        }
                        retired.emplace_back(
                            camera->camera_id, std::move(camera->capture));
                        camera = g_active_cameras.erase(camera);
                    }
                }
                for (auto& [camera_id, capture] : retired) {
                    retireCapture(camera_id, std::move(capture), true);
                }
                try {
                    pqxx::connection conn(DB_CONN_STR);
                    pqxx::work txn(conn);
                    txn.exec("UPDATE cameras SET ai_enabled = false WHERE is_active = true");
                    txn.commit();
                } catch (const std::exception& error) {
                    std::cerr << "[AI STATE WARNING] Could not persist disable-all: " << error.what() << '\n';
                }
            } else {
                std::vector<std::pair<std::string, std::string>> cameras;
                for (auto row : r) {
                    std::string cam_id = row["camera_id"].as<std::string>();
                    std::string name = row["name"].is_null() ? "" : row["name"].as<std::string>();
                    cameras.push_back({cam_id, name});
                }
                changed = static_cast<int>(cameras.size());
                std::thread([cameras, quality, batch_generation]() {
                    for (const auto& [cam_id, name] : cameras) {
                        if (!g_ai_enabled.load() ||
                            g_ai_batch_generation.load() != batch_generation) {
                            break;
                        }
                        std::string stream_id = (quality == "hd") ? cam_id + "_hd" : cam_id;
                        setAICameraState(cam_id, name, stream_id, true);
                        std::this_thread::sleep_for(std::chrono::milliseconds(25));
                    }
                }).detach();
            }
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }

        crow::json::wvalue res;
        res["enabled"] = enabled;
        res["count"] = changed;
        res["scheduled"] = enabled;
        return crow::response(200, res);
    });

    // Endpoint quan ly AI Models
    CROW_ROUTE(app, "/api/ai/models").methods("GET"_method)([]() {
        crow::json::wvalue res;
        res["models_dir"] = "models";
        std::vector<crow::json::wvalue> files;
        try {
            for (const auto& entry : std::filesystem::directory_iterator("models")) {
                if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
                    crow::json::wvalue file;
                    file["name"] = entry.path().filename().string();
                    file["size"] = static_cast<double>(entry.file_size());
                    file["detector_compatible"] = true;
                    file["face_embedding_compatible"] = true;
                    files.push_back(std::move(file));
                }
            }
        } catch (...) {}
        res["files"] = std::move(files);
        
        crow::json::wvalue selected;
        {
            std::lock_guard<std::mutex> lock(g_ai_model_mutex);
            selected["vehicle"] = g_ai_model_vehicle;
            selected["plate"] = g_ai_model_plate;
            selected["face"] = g_ai_model_face;
            selected["face_embedder"] = "face_recognition_sface_2021dec.onnx";
        }
        res["selected"] = std::move(selected);
        return crow::response(res);
    });

    CROW_ROUTE(app, "/api/ai/models/select").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");
        
        std::lock_guard<std::mutex> lock(g_ai_model_mutex);
        if (body.has("vehicle")) g_ai_model_vehicle = body["vehicle"].s();
        if (body.has("plate")) g_ai_model_plate = body["plate"].s();
        if (body.has("face")) g_ai_model_face = body["face"].s();

        crow::json::wvalue res;
        res["ok"] = true;
        return crow::response(200, res);
    });

    CROW_ROUTE(app, "/api/ai/models/open").methods("POST"_method)([]() {
        // Open models folder in Windows explorer
        std::system("explorer.exe models");
        crow::json::wvalue res;
        res["ok"] = true;
        return crow::response(200, res);
    });

    CROW_ROUTE(app, "/api/ai/cache/clear").methods("POST"_method)([]() {
        crow::json::wvalue res;
        res["detections"] = 0;
        res["plate_dedup"] = 0;
        res["object_tracks"] = 0;
        res["object_cache"] = 0;
        return crow::response(200, res);
    });

    CROW_ROUTE(app, "/api/ai/test-frame").methods("POST"_method)([](const crow::request& req) {
        // Mock response for test frame
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");
        
        crow::json::wvalue res;
        res["camera_id"] = body.has("camera_id") ? std::string(body["camera_id"].s()) : std::string("");
        res["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        res["boxes"] = std::vector<crow::json::wvalue>();
        res["results"] = std::vector<crow::json::wvalue>();
        res["processing_ms"] = 0;
        res["latency_ms"] = 0;
        res["frame_seq"] = 1;
        
        return crow::response(200, res);
    });

    // Endpoint thá»‘ng kĂª sá»‘ lÆ°á»£ng sá»± kiá»‡n tá»«ng loáº¡i
    CROW_ROUTE(app, "/api/events/stats").methods("GET"_method)([]() {
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::nontransaction txn(conn);
            pqxx::result r = txn.exec("SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type");
            
            crow::json::wvalue stats;
            stats["vehicle_detected"] = 0;
            stats["person_detected"] = 0;
            stats["face_recognized"] = 0;
            stats["plate_recognized"] = 0;

            for (auto row : r) {
                std::string type = row["event_type"].as<std::string>();
                int count = row["cnt"].as<int>();
                stats[type] = count;
            }
            return crow::response(stats);
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint truy váº¥n thá»‘ng kĂª AI
    CROW_ROUTE(app, "/api/ai_settings").methods("GET"_method)([]() {
        crow::json::wvalue res;
        res["vehicle_det_threshold"] = g_vehicle_det_threshold.load();
        res["plate_det_threshold"] = g_plate_det_threshold.load();
        res["face_det_threshold"] = g_face_det_threshold.load();
        res["plate_threshold"] = g_plate_ocr_threshold.load();
        res["face_threshold"] = g_face_match_threshold.load();
        return res;
    });

    CROW_ROUTE(app, "/api/ai_settings").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");
        if (body.has("vehicle_det_threshold")) g_vehicle_det_threshold.store(std::clamp(static_cast<float>(body["vehicle_det_threshold"].d()), 0.01F, 0.95F));
        if (body.has("plate_det_threshold")) g_plate_det_threshold.store(std::clamp(static_cast<float>(body["plate_det_threshold"].d()), 0.01F, 0.95F));
        if (body.has("face_det_threshold")) g_face_det_threshold.store(std::clamp(static_cast<float>(body["face_det_threshold"].d()), 0.01F, 0.95F));
        if (body.has("plate_threshold")) g_plate_ocr_threshold.store(std::clamp(static_cast<float>(body["plate_threshold"].d()), 0.01F, 0.95F));
        if (body.has("face_threshold")) g_face_match_threshold.store(std::clamp(static_cast<float>(body["face_threshold"].d()), 0.01F, 0.95F));
        return crow::response(200, "Settings updated");
    });

    CROW_ROUTE(app, "/api/ai_roi/<string>").methods("GET"_method)([](const std::string& cam_id) {
        AiRoiConfig roi = getAiRoiConfig(cam_id);
        crow::json::wvalue res;
        res["enabled"] = roi.enabled;
        std::vector<crow::json::wvalue> pts;
        for (const auto& p : roi.points) {
            crow::json::wvalue pt; pt["x"] = p.x; pt["y"] = p.y;
            pts.push_back(std::move(pt));
        }
        res["points"] = std::move(pts);
        return res;
    });

    CROW_ROUTE(app, "/api/ai_roi/<string>").methods("POST"_method)([](const crow::request& req, const std::string& cam_id) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");
        
        AiRoiConfig roi;
        if (body.has("enabled")) roi.enabled = body["enabled"].b();
        if (body.has("points")) {
            for (const auto& p : body["points"]) {
                if (p.has("x") && p.has("y")) {
                    roi.points.push_back({static_cast<float>(p["x"].d()), static_cast<float>(p["y"].d())});
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(g_roi_mutex);
            g_ai_roi_cache[cam_id] = roi;
        }
        return crow::response(200, "ROI updated");
    });

    CROW_ROUTE(app, "/api/ai/stats").methods("GET"_method)([]() {
        crow::json::wvalue stats;
        size_t ai_count = 0;
        {
            std::lock_guard<std::mutex> lock(g_ai_list_mutex);
            ai_count = g_ai_camera_list.size();
        }
        stats["enabled"] = g_ai_enabled.load();
        stats["ai_cameras"] = static_cast<int>(ai_count);
        stats["total_vehicles"] = g_ai_stats.total_vehicles.load();
        stats["total_plates"] = g_ai_stats.total_plates.load();
        stats["total_faces"] = g_ai_stats.total_faces.load();
        stats["total_persons"] = g_ai_stats.total_persons.load();
        stats["frames_processed"] = g_ai_stats.frames_processed.load();
        stats["motion_frames_checked"] = g_ai_stats.motion_frames_checked.load();
        stats["motion_frames_triggered"] = g_ai_stats.motion_frames_triggered.load();
        stats["motion_frames_skipped"] = g_ai_stats.motion_frames_skipped.load();
        stats["motion_frames_queued"] = g_ai_stats.motion_frames_queued.load();
        stats["plate_attempts"] = g_ai_stats.plate_attempts.load();
        stats["plate_boxes_detected"] =
            g_ai_stats.plate_boxes_detected.load();
        stats["plate_retry_successes"] =
            g_ai_stats.plate_retry_successes.load();
        return crow::response(stats);
    });

    // Route phá»¥c vá»¥ giao diá»‡n Web tÄ©nh manager.html
    CROW_ROUTE(app, "/manager.html")([](const crow::request& req, crow::response& res) {
        std::string target_file = "";
        if (std::ifstream("web/manager.html").good()) {
            target_file = "web/manager.html";
        } else if (std::ifstream("../../web/manager.html").good()) {
            target_file = "../../web/manager.html";
        } else if (std::ifstream("../web/manager.html").good()) {
            target_file = "../web/manager.html";
        }

        if (!target_file.empty()) {
            std::ifstream file(target_file, std::ios::binary);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                res.set_header("Content-Type", "text/html; charset=utf-8");
                res.write(buffer.str());
            } else {
                res.code = 404;
                res.write("Khong the mo tep tin manager.html.");
            }
        } else {
            res.code = 404;
            res.write("Khong tim thay tep tin manager.html cua giao dien Web.");
        }
        res.end();
    });

    // Route phá»¥c vá»¥ hĂ¬nh áº£nh snapshot sá»± kiá»‡n AI lá»“ng trong thÆ° má»¥c detections
    CROW_ROUTE(app, "/snapshots/detections/<string>")([](const crow::request& req, crow::response& res, std::string filename) {
        std::string path = "./snapshots/detections/" + filename;
        if (std::ifstream(path).good()) {
            std::ifstream file(path, std::ios::binary);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                res.set_header("Content-Type", "image/jpeg");
                res.write(buffer.str());
            } else {
                res.code = 404;
                res.write("Snapshot file cannot be opened");
            }
        } else {
            res.code = 404;
            res.write("Snapshot not found");
        }
        res.end();
    });

    // Endpoint láº¥y danh sĂ¡ch danh tĂ­nh AI Ä‘Ă£ Ä‘Äƒng kĂ½
    CROW_ROUTE(app, "/api/identities").methods("GET"_method)([](const crow::request& req) {
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::nontransaction txn(conn);
            pqxx::result r = txn.exec("SELECT id, identity_type, label, data_path, is_whitelisted FROM identities ORDER BY id DESC");
            std::vector<crow::json::wvalue> id_json;
            for (auto row : r) {
                crow::json::wvalue val;
                val["id"] = row["id"].as<int>();
                val["identity_type"] = row["identity_type"].as<std::string>();
                val["label"] = row["label"].as<std::string>();
                val["data_path"] = row["data_path"].as<std::string>();
                val["is_whitelisted"] = row["is_whitelisted"].as<bool>();
                id_json.push_back(std::move(val));
            }
            return crow::response(crow::json::wvalue(id_json));
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint xĂ³a danh tĂ­nh AI theo ID
    CROW_ROUTE(app, "/api/identities/<int>").methods("DELETE"_method)([](int id) {
        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("delete_identity", "DELETE FROM identities WHERE id = $1");
            txn.exec_prepared("delete_identity", id);
            txn.commit();
            loadFaceIdentitiesCache();
            return crow::response(200, "Identity deleted successfully");
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Endpoint kiá»ƒm tra tĂ i nguyĂªn há»‡ thá»‘ng vĂ  sá»©c khÂe NVR
    CROW_ROUTE(app, "/api/system/health")([]() {
        try {
            crow::json::wvalue health;
            double used = 0.0;
            double max_storage = 20.0;
            if (g_nvr_engine) {
                used = g_nvr_engine->getStorageUsedGB();
            }
            health["storage_used_gb"] = used;
            health["storage_max_gb"] = max_storage;
            
            // Âáº¿m sá»‘ camera Ä‘ang hoáº¡t Ä‘á»™ng
            size_t active_count = 0;
            {
                std::lock_guard<std::mutex> lock(g_cameras_mutex);
                active_count = g_active_cameras.size();
            }
            health["active_cameras"] = active_count;
            size_t ai_count = 0;
            {
                std::lock_guard<std::mutex> lock(g_ai_list_mutex);
                ai_count = g_ai_camera_list.size();
            }
            const auto runtime = getAiRuntimeSummary();
            health["ai_enabled"] = g_ai_enabled.load();
            health["ai_engine"] = "cpp-onnxruntime";
            health["ai_cameras"] = static_cast<int>(ai_count);
            health["ai_active_cameras"] = runtime.active_cameras;
            health["ai_live_boxes"] = runtime.live_boxes;
            health["ai_last_frame_age_seconds"] = runtime.last_frame_age_seconds;
            health["ai_frames_processed"] = g_ai_stats.frames_processed.load();
            health["ai_motion_enabled"] = g_ai_motion_enabled.load();
            health["ai_motion_frames_checked"] = g_ai_stats.motion_frames_checked.load();
            health["ai_motion_frames_triggered"] = g_ai_stats.motion_frames_triggered.load();
            health["ai_motion_frames_skipped"] = g_ai_stats.motion_frames_skipped.load();
            health["ai_motion_frames_queued"] = g_ai_stats.motion_frames_queued.load();
            health["ai_plate_attempts"] = g_ai_stats.plate_attempts.load();
            health["ai_plate_boxes_detected"] =
                g_ai_stats.plate_boxes_detected.load();
            health["ai_plate_retry_successes"] =
                g_ai_stats.plate_retry_successes.load();
            health["ai_total_vehicles"] = g_ai_stats.total_vehicles.load();
            health["ai_total_plates"] = g_ai_stats.total_plates.load();
            health["ai_total_faces"] = g_ai_stats.total_faces.load();
            health["ai_total_persons"] = g_ai_stats.total_persons.load();

            SystemResourceStats resources;
            {
                std::lock_guard<std::mutex> lock(g_resources_mutex);
                resources = g_cached_resources;
            }
            health["cpu_percent"] = resources.cpu_percent;
            health["ram_used_gb"] = resources.ram_used_gb;
            health["ram_total_gb"] = resources.ram_total_gb;
            health["ram_percent"] = resources.ram_percent;
            health["gpu_name"] = resources.gpu_name;
            health["gpu_percent"] = resources.gpu_percent;
            health["gpu_memory_used_mb"] = resources.gpu_memory_used_mb;
            health["gpu_memory_total_mb"] = resources.gpu_memory_total_mb;
            return crow::response(health);
        } catch (const std::exception& e) {
            return crow::response(500, e.what());
        }
    });

    // Route phá»¥c vá»¥ giao diá»‡n Web tÄ©nh trá»±c tiáº¿p tá»« C++ Server
    CROW_ROUTE(app, "/")([](const crow::request& req, crow::response& res) {
        std::string target_file = "";
        if (std::ifstream("web/index.html").good()) {
            target_file = "web/index.html";
        } else if (std::ifstream("../../web/index.html").good()) {
            target_file = "../../web/index.html";
        } else if (std::ifstream("../web/index.html").good()) {
            target_file = "../web/index.html";
        }

        if (!target_file.empty()) {
            std::ifstream file(target_file, std::ios::binary);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                res.set_header("Content-Type", "text/html; charset=utf-8");
                res.write(buffer.str());
            } else {
                res.code = 404;
                res.write("Khong the mo tep tin index.html.");
            }
        } else {
            res.code = 404;
            res.write("Khong tim thay tep tin index.html cua giao dien Web.");
        }
        res.end();
    });


    // Route phá»¥c vá»¥ hĂ¬nh áº£nh snapshot sá»± kiá»‡n AI
    CROW_ROUTE(app, "/snapshots/<string>")([](const crow::request& req, crow::response& res, std::string filename) {
        std::string path = "./snapshots/" + filename;
        std::string target = "";
        if (std::ifstream(path).good()) {
            target = path;
        } else if (std::ifstream("../../snapshots/" + filename).good()) {
            target = "../../snapshots/" + filename;
        } else if (std::ifstream("../snapshots/" + filename).good()) {
            target = "../snapshots/" + filename;
        }

        if (!target.empty()) {
            std::ifstream file(target, std::ios::binary);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                res.set_header("Content-Type", "image/jpeg");
                res.write(buffer.str());
            } else {
                res.code = 404;
                res.write("Snapshot file cannot be opened");
            }
        } else {
            res.code = 404;
            res.write("Snapshot not found");
        }
        res.end();
    });

    // Backend-owned live stream descriptors.
    CROW_ROUTE(app, "/api/streams/<string>").methods("GET"_method)([](const crow::request& req, const std::string& camera_id) {
        const char* host_header = req.get_header_value("Host").c_str();
        std::string host = host_header && *host_header ? host_header : "127.0.0.1:8080";
        const auto colon = host.find(':');
        const std::string server = colon == std::string::npos ? host : host.substr(0, colon);
        crow::json::wvalue result;
        result["camera_id"] = camera_id;
        result["engine"] = "go2rtc";
        result["rtsp_url"] = "rtsp://" + server + ":8554/" + camera_id;
        result["hls_url"] = "http://" + server + ":1984/api/stream.m3u8?src=" + urlEncode(camera_id) + "&mp4";
        result["mp4_url"] = "http://" + server + ":1984/api/stream.mp4?src=" + urlEncode(camera_id);
        result["webrtc_url"] = "http://" + server + ":1984/api/webrtc?src=" + urlEncode(camera_id);
        return crow::response(result);
    });

    CROW_ROUTE(app, "/api/recordings").methods("GET"_method)([](const crow::request& req) {
        const std::string camera_filter = req.url_params.get("camera_id") ? req.url_params.get("camera_id") : "";
        std::vector<crow::json::wvalue> items;
        const std::filesystem::path root{"recordings"};
        if (std::filesystem::exists(root)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
                if (!entry.is_regular_file() || entry.path().extension() != ".mp4") continue;
                const auto relative = std::filesystem::relative(entry.path(), root).generic_string();
                const auto first_slash = relative.find('/');
                const auto camera_id = first_slash == std::string::npos ? std::string{} : relative.substr(0, first_slash);
                if (!camera_filter.empty() && camera_id != camera_filter) continue;
                crow::json::wvalue item;
                item["camera_id"] = camera_id;
                item["path"] = relative;
                item["size_bytes"] = static_cast<int64_t>(entry.file_size());
                item["playback_url"] = "/api/recordings/file?path=" + urlEncode(relative);
                items.push_back(std::move(item));
            }
        }
        return crow::response(crow::json::wvalue(items));
    });

    CROW_ROUTE(app, "/api/recordings/file").methods("GET"_method)([](const crow::request& req, crow::response& res) {
        const char* raw_path = req.url_params.get("path");
        if (!raw_path) {
            res.code = 400;
            res.end("Missing path");
            return;
        }
        const auto root = std::filesystem::weakly_canonical(std::filesystem::path{"recordings"});
        const auto target = std::filesystem::weakly_canonical(root / std::filesystem::path{raw_path});
        const auto root_text = root.generic_string();
        const auto target_text = target.generic_string();
        if (target_text.rfind(root_text + "/", 0) != 0 || target.extension() != ".mp4" || !std::filesystem::is_regular_file(target)) {
            res.code = 404;
            res.end("Recording not found");
            return;
        }
        res.set_header("Accept-Ranges", "bytes");
        res.set_static_file_info_unsafe(target.string(), "video/mp4");
        res.end();
    });

    CROW_ROUTE(app, "/api/playback/sessions").methods("POST"_method)([](const crow::request& req) {
        const auto body = crow::json::load(req.body);
        if (!body || !body.has("path")) return crow::response(400, "Missing path");
        const std::string relative = body["path"].s();
        const auto root = std::filesystem::weakly_canonical(std::filesystem::path{"recordings"});
        const auto target = std::filesystem::weakly_canonical(root / std::filesystem::path{relative});
        if (target.generic_string().rfind(root.generic_string() + "/", 0) != 0 || !std::filesystem::is_regular_file(target)) {
            return crow::response(404, "Recording not found");
        }
        const auto stream_id = "playback_" + std::to_string(std::hash<std::string>{}(target.string()));
        const auto source = "ffmpeg:" + target.generic_string();
#ifdef _WIN32
        const auto command = "curl.exe -s --connect-timeout 2 --max-time 5 -X PUT -G \"http://127.0.0.1:1984/api/streams\" --data-urlencode \"name=" + stream_id + "\" --data-urlencode \"src=" + source + "\"";
#else
        const auto command = "curl -s --connect-timeout 2 --max-time 5 -X PUT -G \"http://127.0.0.1:1984/api/streams\" --data-urlencode \"name=" + stream_id + "\" --data-urlencode \"src=" + source + "\"";
#endif
        if (system(command.c_str()) != 0) return crow::response(502, "go2rtc registration failed");
        crow::json::wvalue result;
        result["stream_id"] = stream_id;
        result["rtsp_url"] = "rtsp://127.0.0.1:8554/" + stream_id;
        result["hls_url"] = "http://127.0.0.1:1984/api/stream.m3u8?src=" + stream_id + "&mp4";
        result["mp4_url"] = "http://127.0.0.1:1984/api/stream.mp4?src=" + stream_id;
        result["webrtc_url"] = "http://127.0.0.1:1984/api/webrtc?src=" + stream_id;
        return crow::response(201, result);
    });

    // =========================================================================
    // EXTERNAL INTEGRATIONS
    // =========================================================================
    
    // GÂi hĂ m thiáº¿t láº­p cĂ¡c API cho Hikvision & Dahua ANPR tá»« module riĂªng
    setupAnprRoutes(app);

    // Cháº¡y API server trĂªn cá»•ng 8080
    std::cout << "[VMS CORE SERVER] REST API dang chay tren cong 8080..." << std::endl;
    try {
        app.port(8080).multithreaded().run();
    } catch(const std::exception& e) {
        std::cerr << "[CRASH] Crow exception: " << e.what() << std::endl;
    } catch(...) {
        std::cerr << "[CRASH] Unknown exception in Crow run!" << std::endl;
    }

    // DÂn dáº¹p luá»“ng khi thoĂ¡t
    g_running = false;
    if (ai_thread.joinable()) {
        ai_thread.join();
    }
    if (frame_publisher_thread.joinable()) {
        frame_publisher_thread.join();
    }
    if (recording_thread.joinable()) {
        recording_thread.join();
    }
    std::vector<std::pair<std::string, std::unique_ptr<RTSPCapture>>> retired;
    {
        std::lock_guard<std::mutex> lock(g_cameras_mutex);
        for (auto& camera : g_active_cameras) {
            retired.emplace_back(camera.camera_id, std::move(camera.capture));
        }
        g_active_cameras.clear();
    }
    for (auto& [camera_id, capture] : retired) {
        retireCapture(camera_id, std::move(capture), true);
    }
    g_capture_cleanup_shutdown.store(true);
    g_retired_captures_cv.notify_all();
    if (capture_cleanup_thread.joinable()) {
        capture_cleanup_thread.join();
    }
    // Dá»«ng sidecar go2rtc
    stopStatusPoller();
#ifdef _WIN32
    if (sidecar_watchdog_thread.joinable()) sidecar_watchdog_thread.join();
    stopHiddenProcess(g_pi_ocr);
    stopHiddenProcess(g_pi_go2rtc);
    if (g_sidecar_job) CloseHandle(g_sidecar_job);
    if (g_single_instance_mutex) {
        ReleaseMutex(g_single_instance_mutex);
        CloseHandle(g_single_instance_mutex);
    }
#else
    stopOCRService();
    stopGo2RTCSidecar();
#endif
    return 0;
}
