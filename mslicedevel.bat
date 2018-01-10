::
:: Basic wrapper script for running in development mode.
:: It assumes that mantid is installed in the standard
:: location.
::
@set THIS_DIR=%~dp0
@set PYTHON_EXE=C:\MantidInstall\bin\python.exe
@set QT_API=pyqt

:: Build and run mslice
%PYTHON_EXE% %THIS_DIR%setup.py build_py --inplace
@if %ERRORLEVEL% NEQ 0 (
  exit /B %ERRORLEVEL%
)
%THIS_DIR%\scripts\mslice.bat
