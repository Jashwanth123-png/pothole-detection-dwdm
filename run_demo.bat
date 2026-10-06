@echo off
echo ============================================================
echo  REAL-TIME POTHOLE DETECTION - DWDM PROJECT
echo  DEMO MODE LAUNCHER
echo ============================================================
echo.
echo This will:
echo   1. Setup project directories
echo   2. Initialize the database
echo   3. Seed DEMO data (NOT RDD2022 data)
echo   4. Launch the Streamlit dashboard
echo.
echo NOTE: This is DEMO MODE. Real results require:
echo   - RDD2022 dataset placed in dataset/rdd2022/
echo   - A trained YOLO model at models/pothole_yolo.pt
echo.
pause
echo.
echo Setting up project...
python setup_project.py
echo.
echo Launching dashboard...
streamlit run app.py
pause
