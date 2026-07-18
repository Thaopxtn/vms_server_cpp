#include "AnprIntegration.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <unordered_set>
#include <mutex>
#include <pqxx/pqxx>

// External globals from main.cpp
extern const std::string DB_CONN_STR;
extern std::unordered_set<crow::websocket::connection*> g_ws_connections;
extern std::mutex g_ws_mutex;

void setupAnprRoutes(crow::App<CORSMiddleware>& app) {
    // =========================================================================
    // HIKVISION ANPR ISAPI INTEGRATION
    // =========================================================================

    CROW_ROUTE(app, "/api/hikvision/anpr_event").methods("POST"_method)([](const crow::request& req) {
        std::string body = req.body;
        std::string licensePlate = "";
        
        size_t start_lp = body.find("<licensePlate>");
        size_t end_lp = body.find("</licensePlate>");
        if (start_lp != std::string::npos && end_lp != std::string::npos) {
            licensePlate = body.substr(start_lp + 14, end_lp - start_lp - 14);
            licensePlate.erase(0, licensePlate.find_first_not_of(" \t\r\n"));
            licensePlate.erase(licensePlate.find_last_not_of(" \t\r\n") + 1);
        }

        if (licensePlate.empty()) {
            return crow::response(400, "No license plate found in event.");
        }

        std::string jpeg_data = "";
        size_t jpeg_start = body.find("\xFF\xD8");
        if (jpeg_start != std::string::npos) {
            size_t jpeg_end = body.find("\xFF\xD9", jpeg_start);
            if (jpeg_end != std::string::npos) {
                jpeg_data = body.substr(jpeg_start, jpeg_end - jpeg_start + 2);
            }
        }

        std::string snapshot_filename = "";
        if (!jpeg_data.empty()) {
            std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
            snapshot_filename = "hikvision_" + licensePlate + "_" + timestamp + ".jpg";
            std::ofstream out("snapshots/" + snapshot_filename, std::ios::binary);
            if (out.is_open()) {
                out.write(jpeg_data.data(), jpeg_data.size());
                out.close();
            }
        }

        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("ins_hik_det", "INSERT INTO detections (camera_id, object_type, confidence, bounding_box, snapshot_url, attributes) VALUES ($1, $2, $3, $4, $5, $6)");
            crow::json::wvalue attrs;
            attrs["plate"] = licensePlate;
            txn.exec_prepared("ins_hik_det", "HIKVISION_ANPR", "license_plate", 1.0, "[0,0,0,0]", "/snapshots/" + snapshot_filename, attrs.dump());
            txn.commit();
            
            std::string eventMsg = "{\"type\":\"detection\", \"camera_id\":\"HIKVISION_ANPR\", \"object_type\":\"license_plate\", \"snapshot_url\":\"/snapshots/" + snapshot_filename + "\", \"attributes\":{\"plate\":\"" + licensePlate + "\"}}";
            std::lock_guard<std::mutex> lock(g_ws_mutex);
            for (auto* conn_ws : g_ws_connections) {
                conn_ws->send_text(eventMsg);
            }
        } catch (const std::exception& e) {
            std::cerr << "[HIKVISION ERROR] " << e.what() << std::endl;
        }

        return crow::response(200, "Event received");
    });

    CROW_ROUTE(app, "/api/hikvision/config_host").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");
        std::string ip = body["camera_ip"].s();
        std::string user = body["username"].s();
        std::string pass = body["password"].s();
        std::string listening_ip = body["listening_ip"].s();
        std::string listening_port = body["listening_port"].s();
        
        std::string xml = 
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<HttpHostNotificationList>\n"
            "  <HttpHostNotification>\n"
            "    <id>1</id>\n"
            "    <url>/api/hikvision/anpr_event</url>\n"
            "    <protocolType>HTTP</protocolType>\n"
            "    <parameterFormatType>XML</parameterFormatType>\n"
            "    <addressingFormatType>ipaddress</addressingFormatType>\n"
            "    <ipAddress>" + listening_ip + "</ipAddress>\n"
            "    <portNo>" + listening_port + "</portNo>\n"
            "    <httpAuthenticationMethod>none</httpAuthenticationMethod>\n"
            "  </HttpHostNotification>\n"
            "</HttpHostNotificationList>";
            
        std::ofstream out("hik_host.xml");
        out << xml;
        out.close();

        std::string curl_cmd = "curl.exe -s -X PUT --digest -u " + user + ":" + pass + " -d @hik_host.xml http://" + ip + "/ISAPI/Event/notification/httpHosts";
        system(curl_cmd.c_str());
        
        return crow::response(200, "Config command sent");
    });

    CROW_ROUTE(app, "/api/hikvision/sync_list").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");
        std::string ip = body["camera_ip"].s();
        std::string user = body["username"].s();
        std::string pass = body["password"].s();
        
        std::string xml = 
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
            "<LicensePlateAuditData>\n"
            "  <LicensePlateList>\n";
            
        if (body.has("plates")) {
            for (const auto& plate : body["plates"]) {
                std::string list_type = plate["list_type"].s();
                std::string number = plate["number"].s();
                xml += "    <LicensePlate>\n";
                xml += "      <plateNumber>" + number + "</plateNumber>\n";
                xml += "      <group>" + list_type + "</group>\n";
                xml += "    </LicensePlate>\n";
            }
        }
        
        xml += "  </LicensePlateList>\n</LicensePlateAuditData>";
        
        std::ofstream out("hik_list.xml");
        out << xml;
        out.close();

        std::string curl_cmd = "curl.exe -s -X PUT --digest -u " + user + ":" + pass + " -d @hik_list.xml http://" + ip + "/ISAPI/Traffic/channels/1/licensePlateAuditData?fileType=xml";
        system(curl_cmd.c_str());
        
        return crow::response(200, "Sync command sent");
    });

    // =========================================================================
    // DAHUA ANPR CGI INTEGRATION
    // =========================================================================

    CROW_ROUTE(app, "/api/dahua/anpr_event").methods("POST"_method)([](const crow::request& req) {
        std::string body = req.body;
        std::string licensePlate = "";
        
        size_t plate_pos = body.find("\"PlateNumber\"");
        if (plate_pos == std::string::npos) {
            plate_pos = body.find("PlateNumber=");
        }
        
        if (plate_pos != std::string::npos) {
            size_t val_start = plate_pos + 12;
            if (body[plate_pos] == '"') val_start = plate_pos + 14;
            
            while (val_start < body.length() && (body[val_start] == ' ' || body[val_start] == ':' || body[val_start] == '"')) {
                val_start++;
            }
            
            size_t val_end = val_start;
            while (val_end < body.length() && body[val_end] != '"' && body[val_end] != '&' && body[val_end] != '\r' && body[val_end] != '\n' && body[val_end] != ',') {
                val_end++;
            }
            
            if (val_end > val_start) {
                licensePlate = body.substr(val_start, val_end - val_start);
            }
        }

        if (licensePlate.empty()) {
            return crow::response(400, "No license plate found in event.");
        }

        std::string jpeg_data = "";
        size_t jpeg_start = body.find("\xFF\xD8");
        if (jpeg_start != std::string::npos) {
            size_t jpeg_end = body.find("\xFF\xD9", jpeg_start);
            if (jpeg_end != std::string::npos) {
                jpeg_data = body.substr(jpeg_start, jpeg_end - jpeg_start + 2);
            }
        }

        std::string snapshot_filename = "";
        if (!jpeg_data.empty()) {
            std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
            snapshot_filename = "dahua_" + licensePlate + "_" + timestamp + ".jpg";
            std::ofstream out("snapshots/" + snapshot_filename, std::ios::binary);
            if (out.is_open()) {
                out.write(jpeg_data.data(), jpeg_data.size());
                out.close();
            }
        }

        try {
            pqxx::connection conn(DB_CONN_STR);
            pqxx::work txn(conn);
            conn.prepare("ins_dahua_det", "INSERT INTO detections (camera_id, object_type, confidence, bounding_box, snapshot_url, attributes) VALUES ($1, $2, $3, $4, $5, $6)");
            crow::json::wvalue attrs;
            attrs["plate"] = licensePlate;
            txn.exec_prepared("ins_dahua_det", "DAHUA_ANPR", "license_plate", 1.0, "[0,0,0,0]", "/snapshots/" + snapshot_filename, attrs.dump());
            txn.commit();
            
            std::string eventMsg = "{\"type\":\"detection\", \"camera_id\":\"DAHUA_ANPR\", \"object_type\":\"license_plate\", \"snapshot_url\":\"/snapshots/" + snapshot_filename + "\", \"attributes\":{\"plate\":\"" + licensePlate + "\"}}";
            std::lock_guard<std::mutex> lock(g_ws_mutex);
            for (auto* conn_ws : g_ws_connections) {
                conn_ws->send_text(eventMsg);
            }
        } catch (const std::exception& e) {
            std::cerr << "[DAHUA ERROR] " << e.what() << std::endl;
        }

        return crow::response(200, "Event received");
    });

    CROW_ROUTE(app, "/api/dahua/config_host").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");
        std::string ip = body["camera_ip"].s();
        std::string user = body["username"].s();
        std::string pass = body["password"].s();
        std::string listening_ip = body["listening_ip"].s();
        std::string listening_port = body["listening_port"].s();
        
        std::string curl_cmd = "curl.exe -s --digest -u " + user + ":" + pass + 
            " \"http://" + ip + "/cgi-bin/configManager.cgi?action=setConfig&NetApp.Http.Server[0].Enable=true" +
            "&NetApp.Http.Server[0].Address=" + listening_ip + 
            "&NetApp.Http.Server[0].Port=" + listening_port + "\"";
        system(curl_cmd.c_str());
        
        return crow::response(200, "Config command sent");
    });

    CROW_ROUTE(app, "/api/dahua/sync_list").methods("POST"_method)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");
        std::string ip = body["camera_ip"].s();
        std::string user = body["username"].s();
        std::string pass = body["password"].s();
        
        if (body.has("plates")) {
            for (const auto& plate : body["plates"]) {
                std::string list_type = plate["list_type"].s();
                std::string number = plate["number"].s();
                std::string list_name = (list_type == "white") ? "TrafficWhiteList" : "TrafficBlackList";
                
                std::string curl_cmd = "curl.exe -s --digest -u " + user + ":" + pass + 
                    " \"http://" + ip + "/cgi-bin/recordUpdater.cgi?action=insert&name=" + list_name + 
                    "&" + list_name + "[0].PlateNumber=" + number + "\"";
                system(curl_cmd.c_str());
            }
        }
        
        return crow::response(200, "Sync command sent");
    });
}
