::
:: Basic wrapper script for running in development mode.
::
@setlocal
@set THIS_DIR=%~dp0
@set PYTHONPATH=%~dp0; %CONDA_PREFIX%

python %THIS_DIR%\scripts\start_mslice.py
@endlocal