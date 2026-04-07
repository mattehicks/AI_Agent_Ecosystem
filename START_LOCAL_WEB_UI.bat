@echo off
REM Serves only the dashboard files from web\. Point API at VantaBlack in web\api-config.js (or localStorage).
cd /d "%~dp0web"
echo.
echo Dashboard (static):  http://localhost:8080
echo API:                  uncomment window.AI_ECOSYSTEM_API_BASE in web\api-config.js
echo                         (or use localStorage ai_ecosystem_api_base) for remote FastAPI
echo.
echo Do not open index.html via file:// — use this server so API calls work.
echo.
python -m http.server 8080
