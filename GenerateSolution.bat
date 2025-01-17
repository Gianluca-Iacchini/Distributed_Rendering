@echo off

PUSHD %~dp0

SET VSWHERE="%~dp0\vendor\vswhere\vswhere.exe"
SET CMAKE="%~dp0\vendor\cmake-3.29.1-windows-x86_64\bin\cmake.exe"

REM Find the latest version of Visual Studio using vswhere
for /f "tokens=1,2 delims=." %%v in ('%VSWHERE% -latest -property installationVersion -format value') do (
    set VS_VERSION=%%v
)

IF %VS_VERSION% == 17 (
    SET CMAKE_GENERATOR="Visual Studio 17 2022"
    SET CMAKE_BINARY_DIR=build_vs2022
) ELSE IF %VS_VERSION% == 16 (
    SET CMAKE_GENERATOR="Visual Studio 16 2019"
    SET CMAKE_BINARY_DIR=build_vs2019
) ELSE IF %VS_VERSION% == 15 (
    SET CMAKE_GENERATOR="Visual Studio 15 2017"
    SET CMAKE_BINARY_DIR=build_vs2017
) ELSE IF %VS_VERSION% == 14 (
    SET CMAKE_GENERATOR="Visual Studio 14 2015"
    SET CMAKE_BINARY_DIR=build_vs2015
) ELSE (
    ECHO.
    ECHO ***********************************************************************
    ECHO *                                                                     *
    ECHO *                                ERROR                                *
    ECHO *                                                                     *
    ECHO ***********************************************************************
    ECHO No compatible version of Microsoft Visual Studio detected.
    ECHO Please make sure you have Visual Studio 2015 ^(or newer^) and the 
    ECHO "Game Development with C++" workload installed before running this script.
    ECHO. 
    PAUSE
    GOTO :Exit
)

ECHO Updating Git Submodules (this may take a while) ...

git submodule init
git submodule update

ECHO CMake Generator: %CMAKE_GENERATOR%
ECHO CMake Binary Directory: %CMAKE_BINARY_DIR%
ECHO.

MKDIR %CMAKE_BINARY_DIR%
PUSHD %CMAKE_BINARY_DIR%

%CMAKE% -G %CMAKE_GENERATOR% -A x64 "%~dp0"

PAUSE
POPD
:EXIT
POPD