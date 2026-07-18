@echo off
echo ==========================================
echo       VMS SERVER BUILD ^& RUN SCRIPT
echo ==========================================
echo.

echo [1/3] Dang bien dich ma nguon C++...
cmake --build build --config Release
if %errorlevel% neq 0 (
    echo.
    echo [LOI] Bien dich that bai! Vui long kiem tra lai code C++.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Dang dong bo giao dien Web va script AI...
xcopy /E /I /Y "web" "build\Release\web" >nul
copy /Y "*.py" "build\Release" >nul

echo.
echo [3/3] Dang khoi dong VMS Server...
cd build\Release
start vms_server_core.exe

echo.
echo [THANH CONG] He thong da hoat dong! Cua so nay se tu dong dong.
timeout /t 3 >nul
