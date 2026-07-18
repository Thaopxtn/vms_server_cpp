// StorageEngine.cpp - C++ Source cho luồng bắt hình RTSP v ghi hình NVR
#include "StorageEngine.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cctype>
#include <sstream>
#include <unordered_set>
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

RTSPCapture::RTSPCapture(const std::string& camera_id, const std::string& rtsp_url)
    : camera_id(camera_id), rtsp_url(rtsp_url) {}

RTSPCapture::~RTSPCapture() {
    stop();
}

bool RTSPCapture::start() {
    if (running) return true;

    // Khong block luong chinh voi cap.open(). 
    // captureLoop se tu dong open va reconnect trong thread ngam
    running = true;
    cap_thread = std::thread(&RTSPCapture::captureLoop, this);
    std::cout << "[CAMERA] Bat dau luong ket noi ngam cho Camera: " << camera_id << std::endl;
    return true;
}

void RTSPCapture::stop() {
    if (!running) return;

    running = false;
    if (cap_thread.joinable()) {
        cap_thread.join();
    }
    
    if (cap.isOpened()) {
        cap.release();
    }
    opened = false;
    std::cout << "[CAMERA] Da dung doc luong cho Camera: " << camera_id << std::endl;
}

cv::Mat RTSPCapture::getLatestFrame() {
    const auto frame = getLatestFrameShared();
    return frame ? frame->clone() : cv::Mat{};
}

RTSPCapture::FramePtr RTSPCapture::getLatestFrameShared() {
    std::lock_guard<std::mutex> lock(frame_mutex);
    return latest_frame;
}

void RTSPCapture::captureLoop() {
    cv::Mat frame;
    int reconnect_attempts = 0;

    while (running) {
        if (!cap.isOpened()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            if (!running) break;
            std::cout << "[CAMERA] Dang ket noi lai Camera " << camera_id << " (Lan " << ++reconnect_attempts << ")..." << std::endl;
            const std::vector<int> capture_options{
                cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000,
            };
#ifdef _WIN32
            const bool prefer_hardware_decode = std::getenv("VMS_DISABLE_HW_DECODE") == nullptr;
            if (prefer_hardware_decode) {
                const std::vector<int> hardware_options{
                    cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                    cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000,
                    cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_D3D11,
                    cv::CAP_PROP_HW_DEVICE, 0,
                };
                cap.open(rtsp_url, cv::CAP_FFMPEG, hardware_options);
            }
#endif
            if (!cap.isOpened()) cap.open(rtsp_url, cv::CAP_FFMPEG, capture_options);
            if (cap.isOpened()) cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            if (cap.isOpened()) {
                opened = true;
                reconnect_attempts = 0;
            }
            continue;
        }

        // ọc frame (grab + retrieve). Dng grab() lin tục ể dọn bộ ệm frame cũ
        if (cap.grab()) {
            if (cap.retrieve(frame)) {
                if (!frame.empty()) {
                    auto published = std::make_shared<cv::Mat>(std::move(frame));
                    {
                        std::lock_guard<std::mutex> lock(frame_mutex);
                        latest_frame = std::move(published);
                    }
                    frame = cv::Mat{};
                }
            }
        } else {
            // Rớt kết nối
            std::cerr << "[CAMERA ERROR] Mat luong RTSP cho Camera: " << camera_id << std::endl;
            cap.release();
            opened = false;
        }
        
        // Sleep nhỏ ể giảm tải CPU nếu camera gửi frame chậm hn
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// ─── NVR STORAGE ENGINE IMPLEMENTATION ───

NVRStorageEngine::NVRStorageEngine(const std::string& storage_dir, double max_storage_gb)
    : storage_dir(storage_dir), max_storage_gb(max_storage_gb) {
#ifdef _WIN32
    recorder_job = CreateJobObjectA(nullptr, nullptr);
    if (recorder_job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(reinterpret_cast<HANDLE>(recorder_job),
                                     JobObjectExtendedLimitInformation,
                                     &limits, sizeof(limits))) {
            CloseHandle(reinterpret_cast<HANDLE>(recorder_job));
            recorder_job = nullptr;
        }
    }
#endif
    
    // Tạo th mục lu trữ nếu cha tồn tại
    try {
        if (!fs::exists(storage_dir)) {
            fs::create_directories(storage_dir);
        }
    } catch (const std::exception& e) {
        std::cerr << "[NVR ERROR] Khong the tao thu muc luu tru: " << e.what() << std::endl;
    }
}

NVRStorageEngine::~NVRStorageEngine() {
    stopAllStreamRecorders();
#ifdef _WIN32
    if (recorder_job) {
        CloseHandle(reinterpret_cast<HANDLE>(recorder_job));
        recorder_job = nullptr;
    }
#endif
    std::lock_guard<std::mutex> lock(writer_mutex);
    for (auto& [cam_id, state] : active_writers) {
        if (state.writer && state.writer->isOpened()) {
            state.writer->release();
        }
    }
}


namespace {

std::string safeRecorderPathComponent(const std::string& value) {
    std::string safe;
    safe.reserve(value.size());
    for (const unsigned char ch : value) {
        safe.push_back(std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.' ? static_cast<char>(ch) : '_');
    }
    return safe.empty() ? "camera" : safe;
}

std::string quoteRecorderArgument(const std::string& value) {
    return std::string(1, '"') + value + std::string(1, '"');
}

std::string recorderDayDirectory() {
    const auto now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream stream;
    stream << std::setfill('0') << std::setw(4) << local.tm_year + 1900 << '-'
           << std::setw(2) << local.tm_mon + 1 << '-' << std::setw(2) << local.tm_mday;
    return stream.str();
}

#ifdef _WIN32
bool recorderProcessRunning(void* process_handle) {
    const auto process = reinterpret_cast<HANDLE>(process_handle);
    if (!process) return false;
    DWORD exit_code{};
    return GetExitCodeProcess(process, &exit_code) && exit_code == STILL_ACTIVE;
}
#endif

}  // namespace

void NVRStorageEngine::reconcileStreamRecorders(
    const std::vector<std::pair<std::string, std::string>>& camera_streams,
    const std::string& ffmpeg_executable) {
    std::unordered_map<std::string, std::string> desired;
    desired.reserve(camera_streams.size());
    for (const auto& [camera_id, rtsp_url] : camera_streams) desired[camera_id] = rtsp_url;

    std::lock_guard<std::mutex> lock(recorder_mutex);
    std::vector<std::string> obsolete;
    for (const auto& [camera_id, state] : stream_recorders) {
        const auto found = desired.find(camera_id);
#ifdef _WIN32
        const bool stopped = !state || !recorderProcessRunning(state->process_handle);
#else
        const bool stopped = true;
#endif
        if (found == desired.end() || !state || state->rtsp_url != found->second || stopped) {
            obsolete.push_back(camera_id);
        }
    }
    for (const auto& camera_id : obsolete) stopStreamRecorderLocked(camera_id);
    for (const auto& [camera_id, rtsp_url] : desired) {
        if (!stream_recorders.contains(camera_id)) {
            startStreamRecorderLocked(camera_id, rtsp_url, ffmpeg_executable);
        }
    }
}

bool NVRStorageEngine::startStreamRecorderLocked(
    const std::string& camera_id,
    const std::string& rtsp_url,
    const std::string& ffmpeg_executable) {
#ifdef _WIN32
    const auto camera_dir =
        fs::path{storage_dir} / safeRecorderPathComponent(camera_id) / recorderDayDirectory();
    std::error_code error;
    fs::create_directories(camera_dir, error);
    if (error) {
        std::cerr << "[NVR ERROR] Khong the tao thu muc ghi hinh: " << error.message() << '\n';
        return false;
    }

    const auto output_pattern = (camera_dir / "%H-%M-%S.mp4").string();
    std::string command =
        quoteRecorderArgument(ffmpeg_executable) +
        " -hide_banner -loglevel warning -rtsp_transport tcp -fflags +genpts -i " +
        quoteRecorderArgument(rtsp_url) +
        " -map 0:v:0 -c:v copy -an -f segment -segment_time 300 "
        "-reset_timestamps 1 -strftime 1 -segment_format mp4 "
        "-segment_format_options movflags=+frag_keyframe+empty_moov+default_base_moof " +
        quoteRecorderArgument(output_pattern);

    SECURITY_ATTRIBUTES attributes{};
    attributes.nLength = sizeof(attributes);
    attributes.bInheritHandle = TRUE;
    HANDLE stdin_read{nullptr};
    HANDLE stdin_write{nullptr};
    if (!CreatePipe(&stdin_read, &stdin_write, &attributes, 0)) return false;
    SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);

    const HANDLE null_handle = CreateFileA(
        "NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        &attributes, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    startup.wShowWindow = SW_HIDE;
    startup.hStdInput = stdin_read;
    startup.hStdOutput = null_handle;
    startup.hStdError = null_handle;

    PROCESS_INFORMATION process{};
    const BOOL created = CreateProcessA(
        nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED,
        nullptr, nullptr, &startup, &process);
    CloseHandle(stdin_read);
    if (null_handle != INVALID_HANDLE_VALUE) CloseHandle(null_handle);
    if (!created) {
        CloseHandle(stdin_write);
        std::cerr << "[NVR ERROR] FFmpeg stream-copy start failed for " << camera_id
                  << " (" << GetLastError() << ")\n";
        return false;
    }
    if (recorder_job && !AssignProcessToJobObject(
            reinterpret_cast<HANDLE>(recorder_job), process.hProcess)) {
        TerminateProcess(process.hProcess, 0);
        CloseHandle(process.hProcess);
        CloseHandle(process.hThread);
        CloseHandle(stdin_write);
        return false;
    }
    ResumeThread(process.hThread);

    auto state = std::make_unique<StreamRecorderState>();
    state->rtsp_url = rtsp_url;
    state->process_handle = process.hProcess;
    state->thread_handle = process.hThread;
    state->stdin_write = stdin_write;
    state->process_id = process.dwProcessId;
    std::cout << "[NVR] Stream-copy dang ghi " << camera_id
              << " (PID " << state->process_id << ")\n";
    stream_recorders[camera_id] = std::move(state);
    return true;
#else
    (void)camera_id;
    (void)rtsp_url;
    (void)ffmpeg_executable;
    return false;
#endif
}

void NVRStorageEngine::stopStreamRecorderLocked(const std::string& camera_id) {
    const auto found = stream_recorders.find(camera_id);
    if (found == stream_recorders.end()) return;
#ifdef _WIN32
    auto& state = *found->second;
    if (state.stdin_write) {
        DWORD written{};
        const char quit[] = {'q', '\n'};
        const auto input = reinterpret_cast<HANDLE>(state.stdin_write);
        WriteFile(input, quit, sizeof(quit), &written, nullptr);
        FlushFileBuffers(input);
        CloseHandle(input);
        state.stdin_write = nullptr;
    }
    const auto process = reinterpret_cast<HANDLE>(state.process_handle);
    if (process) {
        if (WaitForSingleObject(process, 5000) == WAIT_TIMEOUT) {
            TerminateProcess(process, 0);
            WaitForSingleObject(process, 1000);
        }
        CloseHandle(process);
    }
    const auto thread = reinterpret_cast<HANDLE>(state.thread_handle);
    if (thread) CloseHandle(thread);
#endif
    stream_recorders.erase(found);
}

void NVRStorageEngine::stopAllStreamRecorders() {
    std::lock_guard<std::mutex> lock(recorder_mutex);
    std::vector<std::string> camera_ids;
    camera_ids.reserve(stream_recorders.size());
    for (const auto& [camera_id, state] : stream_recorders) {
        (void)state;
        camera_ids.push_back(camera_id);
    }
    for (const auto& camera_id : camera_ids) stopStreamRecorderLocked(camera_id);
}

void NVRStorageEngine::writeFrame(const std::string& camera_id, const cv::Mat& frame) {
    if (frame.empty()) return;

    std::lock_guard<std::mutex> lock(writer_mutex);
    
    auto& state = active_writers[camera_id];
    time_t now = time(nullptr);
    
    // Ghi hình phn oạn 5 pht (300 giy)
    bool need_new_file = (state.writer == nullptr) || 
                          (!state.writer->isOpened()) || 
                          (now - state.file_start_time >= 300);

    if (need_new_file) {
        if (state.writer && state.writer->isOpened()) {
            state.writer->release();
            std::cout << "[NVR] Da luu xong phan doan: " << state.current_file_path << std::endl;
            
            // Chạy kiểm tra dung lợng ổ cứng sau mỗi lần ng phn oạn
            std::thread(&NVRStorageEngine::enforceStorageLimits, this).detach();
        }
        
        rotateFile(camera_id, frame.cols, frame.rows, 5.0); // Mặc ịnh ghi 15 FPS
    }

    if (state.writer && state.writer->isOpened()) {
        state.writer->write(frame);
        state.frame_count++;
    }
}

void NVRStorageEngine::closeCamera(const std::string& camera_id) {
    std::lock_guard<std::mutex> lock(writer_mutex);
    const auto found = active_writers.find(camera_id);
    if (found == active_writers.end()) return;
    auto& state = found->second;
    if (state.writer && state.writer->isOpened()) {
        state.writer->release();
        std::cout << "[NVR] Da finalize phan doan: " << state.current_file_path << '\n';
    }
    active_writers.erase(found);
}

void NVRStorageEngine::rotateFile(const std::string& camera_id, int width, int height, double fps) {
    auto& state = active_writers[camera_id];
    
    // Tạo cấu trc th mục lu trữ theo cấu trc: storage_dir/camera_id/YYYY-MM-DD/
    time_t now = time(nullptr);
    tm* ltm = localtime(&now);
    
    std::stringstream date_ss;
    date_ss << std::setfill('0') 
            << std::setw(4) << 1900 + ltm->tm_year << "-"
            << std::setw(2) << 1 + ltm->tm_mon << "-"
            << std::setw(2) << ltm->tm_mday;
            
    std::string cam_day_dir = storage_dir + "/" + camera_id + "/" + date_ss.str();
    
    try {
        if (!fs::exists(cam_day_dir)) {
            fs::create_directories(cam_day_dir);
        }
    } catch (const std::exception& e) {
        std::cerr << "[NVR ERROR] Khong the tao thu muc con: " << e.what() << std::endl;
        return;
    }

    std::stringstream file_ss;
    file_ss << cam_day_dir << "/"
            << std::setfill('0')
            << std::setw(2) << ltm->tm_hour << "-"
            << std::setw(2) << ltm->tm_min << "-"
            << std::setw(2) << ltm->tm_sec << ".mp4";

    state.current_file_path = file_ss.str();
    state.file_start_time = now;
    state.frame_count = 0;
    state.writer = std::make_unique<cv::VideoWriter>();

    // Sử dụng mã ha H.264 qua OpenH264 hoặc X264
    int fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4');
    
    // Dự phòng MJPEG nếu hệ thống khng ci ặt x264 encoder
    if (!state.writer->open(state.current_file_path, fourcc, fps, cv::Size(width, height), true)) {
        fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        state.writer->open(state.current_file_path, fourcc, fps, cv::Size(width, height), true);
    }

    std::cout << "[NVR] Bat dau phan doan video moi: " << state.current_file_path << std::endl;
}

void NVRStorageEngine::enforceStorageLimits() {
    double used_gb = getStorageUsedGB();
    std::cout << "[NVR STORAGE] Dung luong dang dung: " << std::fixed << std::setprecision(2) << used_gb << " GB / " << max_storage_gb << " GB" << std::endl;
    
    if (used_gb < max_storage_gb) {
        // Kiểm tra dung lợng trống của ổ ĩa hệ thống
        try {
            fs::space_info tmp = fs::space(storage_dir);
            double total = static_cast<double>(tmp.capacity);
            double available = static_cast<double>(tmp.available);
            double usage_percent = (total - available) / total * 100.0;
            
            if (usage_percent < 95.0) {
                return; // Ổ cứng hệ thống còn nhiều chỗ v cha vợt quota cấu hình
            }
            std::cout << "[NVR STORAGE WARNING] O cung sap day: " << usage_percent << "%! Bat dau don dep..." << std::endl;
        } catch (...) {
            return;
        }
    } else {
        std::cout << "[NVR STORAGE WARNING] Vuot quota cau hinh! Bat dau don dep..." << std::endl;
    }

    // Qut v xa cc file video cũ nhất
    std::vector<std::string> video_files = getSortedVideoFiles();
    
    for (const auto& file_path : video_files) {
        try {
            uintmax_t size = fs::file_size(file_path);
            fs::remove(file_path);
            std::cout << "[NVR STORAGE] Da xoa file video cu: " << file_path << std::endl;
            
            used_gb -= static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);
            
            // Dừng xa khi dung lợng ã hạ xuống dới 90% giới hạn
            if (used_gb < max_storage_gb * 0.9) {
                // Kiểm tra lại ổ ĩa
                fs::space_info tmp = fs::space(storage_dir);
                double total = static_cast<double>(tmp.capacity);
                double available = static_cast<double>(tmp.available);
                if ((total - available) / total * 100.0 < 90.0) {
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[NVR STORAGE ERROR] Khong the xoa file: " << file_path << " | " << e.what() << std::endl;
        }
    }
}

double NVRStorageEngine::getStorageUsedGB() {
    uintmax_t total_bytes = 0;
    try {
        for (const auto& entry : fs::recursive_directory_iterator(storage_dir)) {
            if (entry.is_regular_file()) {
                total_bytes += entry.file_size();
            }
        }
    } catch (...) {}
    return static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
}

std::vector<std::string> NVRStorageEngine::getSortedVideoFiles() {
    struct FileInfo {
        std::string path;
        fs::file_time_type write_time;
    };
    
    std::vector<FileInfo> files;
    try {
        for (const auto& entry : fs::recursive_directory_iterator(storage_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".mp4") {
                files.push_back({entry.path().string(), entry.last_write_time()});
            }
        }
    } catch (...) {}

    // Sắp xếp cc tệp theo thời gian sửa ổi (từ cũ nhất ến mới nhất)
    std::sort(files.begin(), files.end(), [](const FileInfo& a, const FileInfo& b) {
        return a.write_time < b.write_time;
    });

    std::vector<std::string> sorted_paths;
    for (const auto& file : files) {
        sorted_paths.push_back(file.path);
    }
    return sorted_paths;
}
