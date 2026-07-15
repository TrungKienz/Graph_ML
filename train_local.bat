@echo off
echo 🎯 Starting Local Training (Windows)
echo Date: %date% %time%

REM Navigate to Graph_ML directory
cd Graph_ML\src

echo.
echo ==========================================
echo 🎯 STEP 1: ILE ABLATION STUDY
echo ==========================================
python run_ile_experiments.py

echo.
echo ==========================================
echo 🔄 STEP 2: AUGMENTATION EXPERIMENTS  
echo ==========================================
python run_augmentation_experiments.py

echo.
echo ==========================================
echo 📊 STEP 3: RESULTS ANALYSIS
echo ==========================================
python compare_results.py

echo.
echo 🎉 FULL TRAINING COMPLETED!
echo Date: %date% %time%
echo 📄 Check Graph_ML\results\ directory for output files
pause