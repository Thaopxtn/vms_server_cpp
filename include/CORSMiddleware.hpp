#ifndef CORS_MIDDLEWARE_HPP
#define CORS_MIDDLEWARE_HPP

#include "crow.h"

struct CORSMiddleware {
    struct context {};
    void before_handle(crow::request& /*req*/, crow::response& /*res*/, context& /*ctx*/) {
        // No-op
    }
    void after_handle(crow::request& req, crow::response& res, context& /*ctx*/) {
        res.add_header("Access-Control-Allow-Origin", "*");
        res.add_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE");
        res.add_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        
        // Add cache control headers for all GET requests to prevent browser caching of API responses
        if (req.method == crow::HTTPMethod::GET) {
            res.add_header("Cache-Control", "no-store, no-cache, must-revalidate, private");
        }

        // Trả lời phản hồi OPTIONS (Preflight) của trình duyệt ngay lập tức
        if (req.method == static_cast<crow::HTTPMethod>(6)) {
            res.code = 204;
            res.end();
        }
    }
};

#endif // CORS_MIDDLEWARE_HPP
