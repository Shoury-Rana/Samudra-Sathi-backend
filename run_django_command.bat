@echo off
title Django Command Runner (Safe Environment)

echo ==========================================================
echo  CONFIGURING TEMPORARY ENVIRONMENT FOR DJANGO COMMAND
echo ==========================================================
echo This script creates a temporary, isolated environment.
echo Your system settings will be restored automatically.
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

REM --- 4. Run the Django management command ---
echo -> Executing process_raw_reports command...
echo.
python manage.py process_raw_reports || goto cleanup

:cleanup
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
echo -> Command finished.
pause
