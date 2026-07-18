-- init_postgres.sql - Script khởi tạo cơ sở dữ liệu PostgreSQL cho VMS.
-- Loại bỏ sự phụ thuộc vào pgvector để chạy độc lập và ổn định mà không cần cài extension ngoài.

-- 1. Bảng DVRs (Đầu ghi hình) - Định nghĩa trước để tạo khóa ngoại cho Cameras
CREATE TABLE IF NOT EXISTS dvrs (
    id SERIAL PRIMARY KEY,
    dvr_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    host VARCHAR(100) NOT NULL,
    http_port INTEGER DEFAULT 80,
    rtsp_port INTEGER DEFAULT 554,
    username VARCHAR(50) DEFAULT 'admin',
    password VARCHAR(100) NOT NULL,
    brand VARCHAR(50) DEFAULT 'generic',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dvrs_dvr_id ON dvrs(dvr_id);

-- Trigger tự động cập nhật updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE OR REPLACE TRIGGER update_dvrs_updated_at
    BEFORE UPDATE ON dvrs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 2. Bảng Cameras (Có liên kết khóa ngoại với Đầu ghi hình NVR/DVR)
CREATE TABLE IF NOT EXISTS cameras (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(50) UNIQUE NOT NULL,
    dvr_id VARCHAR(50) REFERENCES dvrs(dvr_id) ON DELETE SET NULL, -- Khóa ngoại
    name VARCHAR(100) NOT NULL,
    rtsp_url VARCHAR(500) NOT NULL,
    ip_address VARCHAR(100),
    port INTEGER,
    username VARCHAR(50),
    password VARCHAR(100),
    brand VARCHAR(50),
    is_ip_camera BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    grid_position INTEGER DEFAULT -1,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cameras_camera_id ON cameras(camera_id);
CREATE INDEX IF NOT EXISTS idx_cameras_dvr_id ON cameras(dvr_id);

CREATE OR REPLACE TRIGGER update_cameras_updated_at
    BEFORE UPDATE ON cameras
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 3. Bảng Events (Ghi nhận sự kiện phát hiện của AI)
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(50) NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    description TEXT DEFAULT '',
    confidence DOUBLE PRECISION DEFAULT 0.0,
    snapshot_path VARCHAR(500) DEFAULT '',
    plate_text VARCHAR(20) DEFAULT '',
    plate_image_path VARCHAR(500) DEFAULT '',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE events ADD COLUMN IF NOT EXISTS plate_text VARCHAR(20) DEFAULT '';
ALTER TABLE events ADD COLUMN IF NOT EXISTS plate_image_path VARCHAR(500) DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_events_camera_id ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_plate_text ON events(plate_text);

-- 4. Bảng Identities (Đăng ký biển số / khuôn mặt)
CREATE TABLE IF NOT EXISTS identities (
    id SERIAL PRIMARY KEY,
    identity_type VARCHAR(20) NOT NULL, -- 'face' hoặc 'plate'
    label VARCHAR(100) NOT NULL,
    data_path VARCHAR(500) DEFAULT '',
    is_whitelisted BOOLEAN DEFAULT FALSE,
    face_embedding TEXT, -- Lưu mảng vector đặc trưng dưới dạng chuỗi ngăn cách bởi dấu phẩy
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_identities_type ON identities(identity_type);
CREATE INDEX IF NOT EXISTS idx_identities_label ON identities(label);

-- 5. Bang ket qua AI chi tiet va nhom anh ung vien
CREATE TABLE IF NOT EXISTS detections (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(50) NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    object_type VARCHAR(30) NOT NULL,
    plate_text VARCHAR(20) DEFAULT '',
    plate_confidence FLOAT DEFAULT 0.0,
    identity_label VARCHAR(100) DEFAULT '',
    identity_confidence FLOAT DEFAULT 0.0,
    vehicle_image_path VARCHAR(500) DEFAULT '',
    plate_image_path VARCHAR(500) DEFAULT '',
    face_image_path VARCHAR(500) DEFAULT '',
    bbox_x INTEGER DEFAULT 0,
    bbox_y INTEGER DEFAULT 0,
    bbox_w INTEGER DEFAULT 0,
    bbox_h INTEGER DEFAULT 0,
    location_name VARCHAR(200) DEFAULT '',
    extra_data TEXT DEFAULT ''
);

ALTER TABLE detections ADD COLUMN IF NOT EXISTS extra_data TEXT DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_detections_camera ON detections(camera_id);
CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_plate ON detections(plate_text);
CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(object_type);

-- 6. Vung crop AI theo camera de tang toc va uu tien vung anh tot
CREATE TABLE IF NOT EXISTS ai_camera_rois (
    camera_id VARCHAR(50) PRIMARY KEY REFERENCES cameras(camera_id) ON DELETE CASCADE,
    roi_x DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    roi_y DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    roi_w DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    roi_h DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    roi_points TEXT DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE ai_camera_rois ADD COLUMN IF NOT EXISTS roi_points TEXT DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_ai_camera_rois_camera ON ai_camera_rois(camera_id);
