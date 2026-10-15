@echo off
title Nexus AI Analytics Platform Launcher
echo ===================================================
echo Launching Nexus AI Analytics Platform...
echo 1. Backend Server (http://localhost:8000)
echo 2. Frontend Interface (http://localhost:5173)
echo ===================================================
start "Nexus AI Backend API" cmd /k "python backend/app/main.py"
start "Nexus AI React Frontend" cmd /k "npm run dev"
echo Both servers started in separate terminal windows.
