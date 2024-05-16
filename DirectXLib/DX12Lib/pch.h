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
#include "Commons/Logger.h"
#include "Commons/MathHelper.h"
#include "Commons/Helpers.h"
#include "Commons/CommandContext.h"
#include "Commons/GraphicsCore.h"
#include "DXWrapper/CommandQueue.h"
#include "DXWrapper/CommandList.h"
#include "DXWrapper/CommandAllocator.h"
#include "DXWrapper/DescriptorHeap.h"
#include "DXWrapper/RootSignature.h"
#include "DXWrapper/PipelineState.h"
#include "DXWrapper/SamplerDesc.h"
#include "DXWrapper/Shader.h"
#include "DXWrapper/Texture.h"
#include "DXWrapper/DepthBuffer.h"
#include "DXWrapper/ColorBuffer.h"
#include "DX12Lib/Models/Material.h"
#include "DX12Lib/Models/MaterialManager.h"

#include "DXWrapper/Device.h"