::
:: Basic wrapper script for running in development mode.
::

@setlocal
@set ActiveEnv=false
@set CONDAPATH=C:\Users\eoc57742\AppData\Local\mambaforge
@call %CONDAPATH%\Scripts\activate.bat %CONDAPATH%
@if %CONDA_DEFAULT_ENV% == base (
    @set ActiveEnv=true
    @call %CONDAPATH%\Scripts\activate.bat mantidnightly
)
@if %CONDA_DEFAULT_ENV% == base (
    @echo Please run `conda env create -f mslice-developer.yml` first.
    exit /b 1
)
@set THIS_DIR=%~dp0
@set PYTHONPATH=%~dp0; %CONDA_PREFIX%

python %THIS_DIR%scripts\start_mslice.py
@if %ActiveEnv%==true call conda deactivate
@endlocal