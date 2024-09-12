#include "RayTracingHelpers.h"
#include "DX12Lib/pch.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"


using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;

void CVGI::Octree::CreateOctree(unsigned int maxDepth, unsigned int maxElements)
{
	std::vector<unsigned int> elementsIndices;

	for (unsigned int i = 0; i < m_elementsAABBs.size(); i++)
	{
		elementsIndices.push_back(i);
	}

	m_maxDepth = maxDepth;
	m_maxElements = maxElements;

	CreateOctreeRecursive(DirectX::XMFLOAT3(0, 0, 0), m_OctreeSize, elementsIndices, 0);
}



void CVGI::Octree::CreateOctreeRecursive(DirectX::XMFLOAT3 minBoundary, DirectX::XMFLOAT3 maxBoundary, std::vector<unsigned int> currentIndices, unsigned int currentDepth)
{
	// Subdivide the octree space
	DirectX::XMFLOAT3 offset = DirectX::XMFLOAT3(
		(maxBoundary.x - minBoundary.x) / 2,
		(maxBoundary.y - minBoundary.y) / 2,
		(maxBoundary.z - minBoundary.z) / 2);

	DirectX::XMFLOAT3 newMinBoundary[8];
	DirectX::XMFLOAT3 newMaxBoundary[8];

	for (unsigned int x = 0; x < 2; x++)
	{
		for (unsigned int y = 0; y < 2; y++)
		{
			for (unsigned int z = 0; z < 2; z++)
			{
				unsigned int index = x + 2 * y + 4 * z;
				newMinBoundary[index] = DirectX::XMFLOAT3(
					minBoundary.x + x * offset.x,
					minBoundary.y + y * offset.y,
					minBoundary.z + z * offset.z);
				newMaxBoundary[index] = DirectX::XMFLOAT3(
					newMinBoundary[index].x + offset.x,
					newMinBoundary[index].y + offset.y,
					newMinBoundary[index].z + offset.z);
			}
		}
	}

	std::vector<unsigned int> newIndices[8];

	// Assign elements to the octant that covers the largest portion of the object
	for (unsigned int index : currentIndices)
	{
		auto& element = m_elementsAABBs[index];

		float maxVolume = -10.0f;
		unsigned int bestOctant = 0;

		for (unsigned int i = 0; i < 8; i++)
		{
			// Calculate the overlap volume between the AABB and the octant
			float overlapMinX = max(element.min.x, newMinBoundary[i].x);
			float overlapMaxX = std::min(element.max.x, newMaxBoundary[i].x);
			float overlapMinY = max(element.min.y, newMinBoundary[i].y);
			float overlapMaxY = std::min(element.max.y, newMaxBoundary[i].y);
			float overlapMinZ = max(element.min.z, newMinBoundary[i].z);
			float overlapMaxZ = std::min(element.max.z, newMaxBoundary[i].z);

			// Calculate the extents of the overlap region
			float overlapX = overlapMaxX - overlapMinX;
			float overlapY = overlapMaxY - overlapMinY;
			float overlapZ = overlapMaxZ - overlapMinZ;

			// Ensure overlap is valid (positive extents)
			if (overlapX > 0 && overlapY > 0 && overlapZ > 0)
			{
				float overlapVolume = overlapX * overlapY * overlapZ;

				// Find the octant with the largest overlap volume
				if (overlapVolume > maxVolume)
				{
					maxVolume = overlapVolume;
					bestOctant = i;
				}
			}
		}

		// Assign the element to the octant with the largest overlap
		newIndices[bestOctant].push_back(index);
	}

	// Recursively create child nodes
	for (unsigned int i = 0; i < 8; i++)
	{
		if (newIndices[i].size() > m_maxElements && currentDepth < m_maxDepth)
		{
			CreateOctreeRecursive(newMinBoundary[i], newMaxBoundary[i], newIndices[i], currentDepth + 1);
		}
		else if (!newIndices[i].empty())
		{
			m_elementsPerLeaf.push_back(newIndices[i]);
		}
	}
}

std::vector<std::vector<unsigned int>> CVGI::Octree::GetLeaves()
{
	return m_elementsPerLeaf;
}




void CVGI::AccelerationStructure::Create(UINT byteSize)
{
	Create(1, byteSize);
}

void CVGI::AccelerationStructure::Create(UINT32 elementCount, UINT32 elementSize)
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
}

void CVGI::AccelerationStructure::CreateDerivedViews()
{
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Buffer.NumElements = m_elementCount;
	srvDesc.Buffer.StructureByteStride = m_elementSize;
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

	if (m_srv.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_srv = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	Graphics::s_device->Get()->CreateShaderResourceView(m_resource.Get(), &srvDesc, m_srv);


	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDesc.Format = DXGI_FORMAT_UNKNOWN;
	uavDesc.Buffer.NumElements = m_elementCount;
	uavDesc.Buffer.StructureByteStride = m_elementSize;
	uavDesc.Buffer.CounterOffsetInBytes = 0;
	uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

	if (m_uav.ptr == D3D12_GPU_VIRTUAL_ADDRESS_UNKNOWN)
	{
		m_uav = Graphics::AllocateDescriptor(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	Graphics::s_device->Get()->CreateUnorderedAccessView(m_resource.Get(), nullptr, &uavDesc, m_uav);
}

void CVGI::BottomLevelAccelerationStructure::AddGeometry(UINT count, D3D12_GPU_VIRTUAL_ADDRESS startAddress, size_t stride)
{
	D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;

	geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
	geometryDesc.AABBs.AABBCount = count;
	geometryDesc.AABBs.AABBs.StartAddress = startAddress;
	geometryDesc.AABBs.AABBs.StrideInBytes = stride;
	geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

	m_geometryDescs.push_back(geometryDesc);
}

void CVGI::BottomLevelAccelerationStructure::Build(DX12Lib::CommandContext& context)
{
	assert(m_geometryDescs.size() > 0 && "Error building BLAS with no geometry");

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs = {};

	blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	blasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
	blasInputs.NumDescs = m_geometryDescs.size();
	blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
	blasInputs.pGeometryDescs = m_geometryDescs.data();

	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
	Graphics::s_device->GetDXR()->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputs, &prebuildInfo);

	assert(prebuildInfo.ResultDataMaxSizeInBytes > 0);

	DX12Lib::StructuredBuffer scratchBuffer;
	scratchBuffer.Create(1, prebuildInfo.ScratchDataSizeInBytes);

	this->Create(prebuildInfo.ResultDataMaxSizeInBytes);

	m_instanceDesc.AccelerationStructure = m_gpuVirtualAddress;
	m_instanceDesc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
	m_instanceDesc.InstanceContributionToHitGroupIndex = 0;
	m_instanceDesc.Transform[0][0] = m_instanceDesc.Transform[1][1] = m_instanceDesc.Transform[2][2] = 1.0f;
	m_instanceDesc.InstanceMask = 1;


	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
	buildDesc.Inputs = blasInputs;
	buildDesc.ScratchAccelerationStructureData = scratchBuffer.GetGpuVirtualAddress();
	buildDesc.DestAccelerationStructureData = m_gpuVirtualAddress;

	context.m_commandList->GetDXR()->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

	context.Flush(true);
}

void CVGI::TopLevelAccelerationStructure::AddBLAS(std::shared_ptr<BottomLevelAccelerationStructure> structure)
{
	m_blasVector.push_back(structure);
}

void CVGI::TopLevelAccelerationStructure::Build(DX12Lib::CommandContext& context)
{
	assert(m_blasVector.size() > 0 && "Error building TLAS with no BLAS");

	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};

	tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
	tlasInputs.NumDescs = m_blasVector.size();
	tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;


	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
	Graphics::s_device->GetDXR()->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &prebuildInfo);

	assert(prebuildInfo.ResultDataMaxSizeInBytes > 0);

	DX12Lib::StructuredBuffer scratchBuffer;
	scratchBuffer.Create(1, prebuildInfo.ScratchDataSizeInBytes);

	this->Create(prebuildInfo.ResultDataMaxSizeInBytes);

	std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDescs;

	instanceDescs.resize(m_blasVector.size());

	std::transform(m_blasVector.begin(), m_blasVector.end(), instanceDescs.begin(),
		[](std::shared_ptr<BottomLevelAccelerationStructure> blas) -> D3D12_RAYTRACING_INSTANCE_DESC { return blas->GetInstanceDesc(); });

	DX12Lib::UploadBuffer instanceDescBuffer;
	instanceDescBuffer.Create(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * instanceDescs.size());

	void* mappedData = instanceDescBuffer.Map();
	memcpy(mappedData, instanceDescs.data(), sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * instanceDescs.size());
	instanceDescBuffer.Unmap();

	
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
	tlasInputs.InstanceDescs = instanceDescBuffer.GetGpuVirtualAddress();
	buildDesc.Inputs = tlasInputs;
	buildDesc.ScratchAccelerationStructureData = scratchBuffer.GetGpuVirtualAddress();
	buildDesc.DestAccelerationStructureData = m_gpuVirtualAddress;

	for (auto blas : m_blasVector)
	{
		context.TransitionResource(*blas, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	}

	context.m_commandList->GetDXR()->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

	context.Flush(true);
}
