// StorageEngine.hpp - C++ Header cho luồng bắt hình RTSP v ghi hình NVR
#ifndef STORAGE_ENGINE_HPP
#define STORAGE_ENGINE_HPP

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <unordered_map>

// Lớp chụp hình RTSP chạy ngầm ể trnh tch lũy ộ trễ ệm
class RTSPCapture {
public:
    using FramePtr = std::shared_ptr<const cv::Mat>;

    RTSPCapture(const std::string& camera_id, const std::string& rtsp_url);
    ~RTSPCapture();

    // Bắt ầu luồng ọc camera
    bool start();
    // Dừng luồng ọc camera
    void stop();

    // Lấy frame mới nhất từ camera
    cv::Mat getLatestFrame();
    FramePtr getLatestFrameShared();
    
    bool isOpened() const { return opened; }
    std::string getCameraId() const { return camera_id; }

private:
    void captureLoop();

    std::string camera_id;
    std::string rtsp_url;
    cv::VideoCapture cap;
    
    std::thread cap_thread;
    std::mutex frame_mutex;
    FramePtr latest_frame;
    
    std::atomic<bool> running{false};
    std::atomic<bool> opened{false};
};

// Module NVR ghi hình phn oạn .mp4 v dọn dẹp dung lợng vòng lặp (Ring-Buffer)
class NVRStorageEngine {
public:
    NVRStorageEngine(const std::string& storage_dir, double max_storage_gb = 50.0);
    ~NVRStorageEngine();

    // Ghi nhận frame mới v ghi xuống file phn oạn hiện tại
    void writeFrame(const std::string& camera_id, const cv::Mat& frame);

    // Finalize the active MP4 segment when a camera is released.
    void closeCamera(const std::string& camera_id);

    // Record proxied RTSP streams without decoding/re-encoding video.
    void reconcileStreamRecorders(
        const std::vector<std::pair<std::string, std::string>>& camera_streams,
        const std::string& ffmpeg_executable);
    void stopAllStreamRecorders();

    // Kch hoạt dọn dẹp ĩa khi vợt giới hạn hoặc ĩa ầy (>95%)
    void enforceStorageLimits();

    // Tnh tổng kch thớc th mục
    double getStorageUsedGB();

private:
    struct StreamRecorderState {
        std::string rtsp_url;
        void* process_handle{nullptr};
        void* thread_handle{nullptr};
        void* stdin_write{nullptr};
        unsigned long process_id{0};
    };
    bool startStreamRecorderLocked(const std::string& camera_id,
                                   const std::string& rtsp_url,
                                   const std::string& ffmpeg_executable);
    void stopStreamRecorderLocked(const std::string& camera_id);

    // Mở file phn oạn mới
    void rotateFile(const std::string& camera_id, int width, int height, double fps);
    
    // Qut th mục v lấy danh sch tệp video xếp theo thời gian tng dần
    std::vector<std::string> getSortedVideoFiles();

    std::string storage_dir;
    double max_storage_gb; // Giới hạn lu trữ tối a (n vị GB)
    
    // Quản lý trạng thi ghi của từng camera
    struct WriterState {
        std::unique_ptr<cv::VideoWriter> writer;
        std::string current_file_path;
        time_t file_start_time = 0;
        int frame_count = 0;
    };
    std::unordered_map<std::string, WriterState> active_writers;
    std::mutex writer_mutex;
    std::unordered_map<std::string, std::unique_ptr<StreamRecorderState>> stream_recorders;
    std::mutex recorder_mutex;
    void* recorder_job{nullptr};
};

#endif // STORAGE_ENGINE_HPP
