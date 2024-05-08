#pragma once


// Windows
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h> // For HRESULT

// DirectX
#include <dxgi1_6.h>
#include <directx/d3dx12.h>
#include <DirectXColors.h>
#include <DirectXCollision.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include "DirectXTex.h"


// STD
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <array>
#include <vector>


// Engine
#include "Logger.h"
#include "MathHelper.h"
#include "Helpers.h"
#include "GraphicsCore.h"
#include "Device.h"