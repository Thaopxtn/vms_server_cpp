#ifndef ANPR_INTEGRATION_HPP
#define ANPR_INTEGRATION_HPP

#include "crow.h"

#include "CORSMiddleware.hpp"

// Setup function for ANPR endpoints
void setupAnprRoutes(crow::App<CORSMiddleware>& app);

#endif // ANPR_INTEGRATION_HPP
