::
:: Basic wrapper script for running in development mode.
:: It assumes that mantid is installed in the standard
:: location.
::
@set THIS_DIR=%~dp0
@set PYTHON_EXE=C:\MantidInstall\bin\python.exe

:: Build and run mslice
@set BUILD_DIR=%THIS_DIR%build
%PYTHON_EXE% %THIS_DIR%setup.py build
@if %ERRORLEVEL% NEQ 0 (
  exit /B %ERRORLEVEL%
)
@set PYTHONPATH=%BUILD_DIR%\lib;%PYTHONPATH%
@set PY_VERS=2.7
%BUILD_DIR%\scripts-%PY_VERS%\mslice.bat
