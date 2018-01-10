::
:: Basic wrapper script for running in development mode.
:: It assumes that mantid is installed in the standard
:: location.
::
@set THIS_DIR=%~dp0
@set PYTHON_EXE=C:\MantidInstall\bin\python.exe
@set QT_API=pyqt

)
%THIS_DIR%\scripts\mslice.bat
