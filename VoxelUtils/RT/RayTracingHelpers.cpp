#include "RayTracingHelpers.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "RaytracingStateObject.h"


using namespace VOX;
using namespace DX12Lib;
using namespace Graphics;

void VOX::AccelerationStructure::Create(UINT byteSize)
{
	Create(1, byteSize);
}

void VOX::AccelerationStructure::Create(UINT32 elementCount, UINT32 elementSize)
{
	auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(elementCount * elementSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	ThrowIfFailed(Graphics::s_device->Get()->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&bufferDesc,
		D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
		nullptr,
		IID_PPV_ARGS(m_resource.GetAddressOf())
	));

	m_gpuVirtualAddress = m_resource->GetGPUVirtualAddress();

	CreateDerivedViews();
}

void VOX::AccelerationStructure::CreateDerivedViews()
{
	

	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.RaytracingAccelerationStructure.Location = m_gpuVirtualAddress;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	if (m_srv.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_srv = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	Graphics::s_device->Get()->CreateShaderResourceView(NULL, &srvDesc, m_srv);

}

void VOX::BottomLevelAccelerationStructure::AddGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride, AABB aabb)
{
	m_geometries.push_back(BLASGeometry(count, startAddress, stride, aabb));
}

void VOX::BottomLevelAccelerationStructure::AddGeometry(BLASGeometry& geometry)
{
	m_geometries.push_back(geometry);
}

void VOX::BottomLevelAccelerationStructure::Build(DX12Lib::CommandContext& context)
{
	assert(m_geometries.size() > 0 && "Error building BLAS with no geometry");

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};

	std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs(m_geometries.size());

	for (unsigned int i = 0; i < m_geometries.size(); i++)
	{
		geometryDescs[i] = m_geometries[i].GetGeometryDesc();
	}

	blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
	blasInputs.NumDescs = static_cast<UINT>(geometryDescs.size());
	blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
	blasInputs.pGeometryDescs = geometryDescs.data();

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
	Graphics::s_device->GetDXR()->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputs, &prebuildInfo);

	assert(prebuildInfo.ResultDataMaxSizeInBytes > 0);

	DX12Lib::StructuredBuffer scratchBuffer;
	scratchBuffer.Create(1, prebuildInfo.ScratchDataSizeInBytes);

	this->Create(prebuildInfo.ResultDataMaxSizeInBytes);

	m_instanceDesc = {};
	m_instanceDesc.AccelerationStructure = m_gpuVirtualAddress;
	m_instanceDesc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
	m_instanceDesc.Transform[0][0] = m_instanceDesc.Transform[1][1] = m_instanceDesc.Transform[2][2] = 1.0f;
	m_instanceDesc.InstanceMask = 1;


	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
	buildDesc.Inputs = blasInputs;
	buildDesc.ScratchAccelerationStructureData = scratchBuffer.GetGpuVirtualAddress();
	buildDesc.DestAccelerationStructureData = m_gpuVirtualAddress;

	context.TransitionResource(scratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);

	context.m_commandList->GetDXR()->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

	context.Flush(true);
}

void VOX::TopLevelAccelerationStructure::AddBLAS(std::shared_ptr<BottomLevelAccelerationStructure> structure)
{
	m_blasVector.push_back(structure);
}

void VOX::TopLevelAccelerationStructure::Build(DX12Lib::CommandContext& context)
{
	assert(m_blasVector.size() > 0 && "Error building TLAS with no BLAS");

	std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDescs;

	instanceDescs.resize(m_blasVector.size());

	for (unsigned int i = 0; i < m_blasVector.size(); i++)
	{
		instanceDescs[i] = m_blasVector[i]->GetInstanceDesc();
	}

	size_t instSize = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * instanceDescs.size();

	DX12Lib::UploadBuffer instanceDescBuffer;
	instanceDescBuffer.Create(instSize);

	void* mappedData = instanceDescBuffer.Map();
	memcpy(mappedData, instanceDescs.data(), instSize);
	instanceDescBuffer.Unmap();

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};

	tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
	tlasInputs.NumDescs = m_blasVector.size();
	tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
	tlasInputs.InstanceDescs = instanceDescBuffer.GetGpuVirtualAddress();

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
	Graphics::s_device->GetDXR()->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &prebuildInfo);

	assert(prebuildInfo.ResultDataMaxSizeInBytes > 0);

	DX12Lib::StructuredBuffer scratchBuffer;
	scratchBuffer.Create(1, prebuildInfo.ScratchDataSizeInBytes);

	this->Create(prebuildInfo.ResultDataMaxSizeInBytes);





	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};

	buildDesc.Inputs = tlasInputs;
	buildDesc.ScratchAccelerationStructureData = scratchBuffer.GetGpuVirtualAddress();
	buildDesc.DestAccelerationStructureData = m_gpuVirtualAddress;

	context.TransitionResource(instanceDescBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	context.TransitionResource(scratchBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);

	context.m_commandList->GetDXR()->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

	context.Flush(true);
}

RayTracingContext& VOX::RayTracingContext::Begin()
{
	return reinterpret_cast<RayTracingContext&>(DX12Lib::CommandContext::Begin());
}

void VOX::RayTracingContext::DispatchRays1D(UINT width)
{
	DispatchRays3D(width, 1, 1);
}

void VOX::RayTracingContext::DispatchRays2D(UINT width, UINT height)
{
	DispatchRays3D(width, height, 1);
}

void VOX::RayTracingContext::DispatchRays3D(UINT width, UINT height, UINT depth)
{
	RaytracingStateObject* pipeline = static_cast<RaytracingStateObject*>(m_currentPipelineState);

	if (pipeline != nullptr)
	{
		D3D12_DISPATCH_RAYS_DESC dispatchDesc = pipeline->GetDefaultDispatchDesc();
		dispatchDesc.Width = width;
		dispatchDesc.Height = height;
		dispatchDesc.Depth = depth;

		m_commandList->GetDXR()->DispatchRays(&dispatchDesc);
	}
}

VOX::BLASGeometry::BLASGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride)
	: m_geometryAABB{ DirectX::XMFLOAT3(-0.5f, -0.5f, -0.5f), DirectX::XMFLOAT3(0.5f, 0.5f, 0.5f) }
{
	m_geometryDesc = {};

	m_geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
	m_geometryDesc.AABBs.AABBCount = count;
	m_geometryDesc.AABBs.AABBs.StartAddress = startAddress;
	m_geometryDesc.AABBs.AABBs.StrideInBytes = stride;
	m_geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
}

VOX::BLASGeometry::BLASGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride, AABB& aabb)
	: m_geometryAABB(aabb)
{
	m_geometryDesc = {};

	m_geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
	m_geometryDesc.AABBs.AABBCount = count;
	m_geometryDesc.AABBs.AABBs.StartAddress = startAddress;
	m_geometryDesc.AABBs.AABBs.StrideInBytes = stride;
	m_geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
}

VOX::BLASGeometry::BLASGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride, DirectX::XMFLOAT3 min, DirectX::XMFLOAT3 max)
	: m_geometryAABB{ min, max }
{
	m_geometryDesc = {};

	m_geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
	m_geometryDesc.AABBs.AABBCount = count;
	m_geometryDesc.AABBs.AABBs.StartAddress = startAddress;
	m_geometryDesc.AABBs.AABBs.StrideInBytes = stride;
	m_geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
}

void VOX::BLASGeometry::SetGeometryDesc(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride)
{
	m_geometryDesc = {};

	m_geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
	m_geometryDesc.AABBs.AABBCount = count;
	m_geometryDesc.AABBs.AABBs.StartAddress = startAddress;
	m_geometryDesc.AABBs.AABBs.StrideInBytes = stride;
	m_geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
}
