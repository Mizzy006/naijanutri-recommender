@echo off
echo ==========================================
echo   Starting NaijaNutri Hackathon Demo...
echo ==========================================

:: Start the FastAPI Backend in a new window
echo Starting Backend (Uvicorn)...
start "NaijaNutri Backend" cmd /k ".\venv\Scripts\activate && uvicorn backend.main:app --port 8000"

:: Wait 3 seconds to give the backend a head start
timeout /t 3 /nobreak >nul

:: Start the Streamlit Frontend in a new window
echo Starting Frontend (Streamlit)...
start "NaijaNutri Frontend" cmd /k ".\venv\Scripts\activate && streamlit run frontend/app.py"

echo.
echo All services launched! Check the pop-up windows.
pause