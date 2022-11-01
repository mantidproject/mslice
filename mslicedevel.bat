::
:: Basic wrapper script for running in development mode.
::
@setlocal
@set THIS_DIR=%~dp0

:: Define here the path to your conda installation and uncomment the line
::@set CONDAPATH=

:: ActiveEnv is set to true once the conda environment mantidnightly was successfully activated
@set ActiveEnv=false

:: With CONDAPATH set mantidnightly can be activated even when the base environment is not activated yet, for instance by doubleclicking on mslicedevel.bat
@if defined CONDAPATH (
    goto CheckBaseEnv
) else if defined CONDA_DEFAULT_ENV (
    :: Without CONDAPATH set it is still possible to activate mantidnightly when mslicedevel.bat is run from a conda prompt
    goto ActivateEnv
) else (
    goto NoConda
)

:CheckBaseEnv
@if defined CONDA_DEFAULT_ENV (
    goto ActivateEnv
) else (
    :: Activate the base environment first if necessary
    call %CONDAPATH%\Scripts\activate.bat %CONDAPATH%
    goto ActivateEnv
)

:ActivateEnv
:: Activate mantidnightly
@if %CONDA_DEFAULT_ENV% == base (
    call conda activate mantidnightly
)
@if %CONDA_DEFAULT_ENV% == mantidnightly (
    @set ActiveEnv=true
) else (
    goto NoConda
)

:NoConda
:: This allows to start MSlice when there is no conda available or activating mantidnightly failed
@set MANTIDPATH=C:\MantidInstall\bin
@set PYTHONPATH=%MANTIDPATH%;%THIS_DIR%;%PYTHONPATH%
@set QT_PLUGIN_PATH=%MANTIDPATH%\..\plugins\qt5
%MANTIDPATH%\python.exe %THIS_DIR%scripts\start_mslice.py
@goto End

@set PYTHONPATH=%THIS_DIR%; %CONDA_PREFIX%

python %THIS_DIR%scripts\start_mslice.py

:: Only attempt to deactivate mantidnightly when it was successfully activated
@if %ActiveEnv% == true call conda deactivate

:End
@endlocal