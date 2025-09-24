@echo off
title AI Server Launcher (Safe Environment)

echo ==========================================================
echo  CONFIGURING TEMPORARY ENVIRONMENT FOR AI SERVER
echo ==========================================================
echo This script creates a temporary, isolated environment.
echo Your system settings will be restored when this window is closed.
echo.

REM --- 1. Save original environment variables (if they exist) ---
echo -> Saving original cache locations...
set "OLD_TRANSFORMERS_CACHE=%TRANSFORMERS_CACHE%"
set "OLD_HF_HOME=%HF_HOME%"

REM --- 2. Set new, temporary cache locations on the D: drive ---
echo -> Redirecting caches to D: drive for this session ONLY...
set "TRANSFORMERS_CACHE=D:\sih\prototype\samudra_sathi\ai_model_cache"
set "HF_HOME=D:\sih\prototype\samudra_sathi\ai_model_cache"
echo    New Cache Path: %TRANSFORMERS_CACHE%
echo.

REM --- 3. Activate virtual environment ---
echo -> Activating Python virtual environment...
call .\venv\Scripts\activate
echo.

REM --- 4. Start the AI server ---
echo -> Starting AI Server on port 8001...
echo    (Press CTRL+C here to stop the server)
echo.
cd ai_model
uvicorn endpoint:app --host 0.0.0.0 --port 8001
cd ..

REM --- 5. Cleanup: Restore original environment variables ---
echo.
echo ==========================================================
echo  CLEANING UP AND RESTORING ENVIRONMENT
echo ==========================================================
echo -> Restoring original cache locations...
set "TRANSFORMERS_CACHE=%OLD_TRANSFORMERS_CACHE%"
set "HF_HOME=%OLD_HF_HOME%"

REM Clean up the temporary variables we created to be tidy
set OLD_TRANSFORMERS_CACHE=
set OLD_HF_HOME=

echo -> Environment restored successfully.
echo.
pause