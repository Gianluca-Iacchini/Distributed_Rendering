#include "VoxelMaterial.h"
//#include "DX12Lib/Commons/Renderer.h"

//using namespace Graphics::Renderer;

CVGI::VoxelMaterial::VoxelMaterial()
{
    //m_pso = Graphics::Renderer::s_PSOs[L"PSO_VOXEL"].get();
}

DX12Lib::ConstantBufferMaterial CVGI::VoxelMaterial::BuildMaterialConstantBuffer()
{
    return Material::BuildMaterialConstantBuffer();
}
