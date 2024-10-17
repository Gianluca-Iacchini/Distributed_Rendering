#include "ClusterVisibility.h"
#include "DX12Lib/pch.h"
#include "WinPixEventRuntime/pix3.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "dxcapi.h"
#include "Raytracing.h"
#include "BuildAABBsTechnique.h"
#include "PrefixSumVoxels.h"
#include "ClusterVoxels.h"
#include "FaceCountTechnique.h"
#include "VoxelizeScene.h"

#define MAX_BLAS_COUNT 250

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;


void CVGI::ClusterVisibility::InitializeBuffers()
{
	m_bufferManager->AddStructuredBuffer(m_data->FaceCount, sizeof(DirectX::XMUINT2));
	m_bufferManager->AddStructuredBuffer(1, sizeof(UINT32));

	m_bufferManager->AddStructuredBuffer(1, sizeof(UINT32));
	m_bufferManager->AddStructuredBuffer(1, sizeof(UINT32));

	m_bufferManager->AddByteAddressBuffer();

	m_bufferManager->AllocateBuffers();
}

void CVGI::ClusterVisibility::PerformTechnique(DX12Lib::ComputeContext& computeContext)
{
	computeContext.Flush(true);
	RayTracingContext& context = RayTracingContext::Begin();
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), L"RayTracing");

	UINT side = floor(std::cbrt(m_data->FaceCount));
	DirectX::XMUINT3 dispatchSize = DirectX::XMUINT3(side + 1, side + 1, side);

	m_cbRayTracing.CurrentPhase = 0;
	m_cbRayTracing.DispatchSize = dispatchSize;
	m_cbRayTracing.NumberOfFaces = m_data->FaceCount;
	m_cbRayTracing.GridDimension = m_data->GetVoxelGridSize();
	m_cbRayTracing.Rand1 = MathHelper::RandF();
	m_cbRayTracing.Rand2 = MathHelper::RandF();

	assert(dispatchSize.x * dispatchSize.y * dispatchSize.z >= m_data->FaceCount);

	TechniquePass(context, dispatchSize);

	UINT32 visibleClusterCount = *m_bufferManager->ReadFromBuffer<UINT32*>(context, (UINT)BufferType::ClusterCount);


	m_bufferManager->ResizeBuffer((UINT)BufferType::VisibleCluster, visibleClusterCount);


	m_cbRayTracing.CurrentPhase = 1;
	TechniquePass(context, dispatchSize);

	PIXEndEvent(context.m_commandList->Get());
	context.Finish(true);
}

void CVGI::ClusterVisibility::TechniquePass(RayTracingContext& context, DirectX::XMUINT3 groupSize)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[Name].get());

	auto& aabbBufferManager = m_data->GetBufferManager(BuildAABBsTechnique::Name);
	auto& compactBufferManager = m_data->GetBufferManager(PrefixSumVoxels::Name);
	auto& clusterBufferManager = m_data->GetBufferManager(ClusterVoxels::Name);
	auto& faceBufferManager = m_data->GetBufferManager(FaceCountTechnique::Name);

	m_bufferManager->TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	aabbBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	faceBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)RayTraceRootSignature::RayTraceCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbRayTracing).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::ClusterSRVTable, clusterBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::FaceSRVTable, faceBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::AABBSRVTable, aabbBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootShaderResourceView((UINT)RayTraceRootSignature::AccelerationStructureSRV, m_data->GetTlas()->GetGpuVirtualAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::RayTraceUAVTable, m_bufferManager->GetUAVHandle());
	context.DispatchRays3D(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusterVisibility::BuildRootSignature()
{
	std::shared_ptr<DX12Lib::RootSignature> rayTracingRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)RayTraceRootSignature::Count, 0);

	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::RayTraceCBV].InitAsConstantBuffer(0);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::AccelerationStructureSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::ClusterSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 2);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::FaceSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 3);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::AABBSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 4);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::RayTraceUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 5);


	rayTracingRootSignature->Finalize();

	return rayTracingRootSignature;
}

std::shared_ptr<DX12Lib::PipelineState> CVGI::ClusterVisibility::BuildPipelineState()
{
	std::shared_ptr<DX12Lib::RootSignature> globalRootSig = BuildRootSignature();

	std::shared_ptr<RaytracingStateObject> rayTracingPso = std::make_shared<RaytracingStateObject>();

	auto shaderBlob = CD3DX12_SHADER_BYTECODE((void*)g_pRaytracing, ARRAYSIZE(g_pRaytracing));

	rayTracingPso->SetRootSignature(globalRootSig);
	rayTracingPso->SetShaderBytecode(shaderBlob);
	rayTracingPso->SetShaderEntryPoint(RayTracingShaderType::Raygen, L"MyRaygenShader");
	rayTracingPso->SetShaderEntryPoint(RayTracingShaderType::Miss, L"MyMissShader");
	rayTracingPso->SetShaderEntryPoint(RayTracingShaderType::ClosestHit, L"MyClosestHitShader");
	rayTracingPso->SetShaderEntryPoint(RayTracingShaderType::Intersection, L"MyIntersectionShader");
	rayTracingPso->SetAttributeAndPayloadSize(sizeof(UINT32), sizeof(UINT32));
	rayTracingPso->SetRecursionDepth(1);

	auto& hitGroup = rayTracingPso->CreateHitGroup(L"MyHitGroup");
	hitGroup.AddClosestHitShader(L"MyClosestHitShader");
	hitGroup.AddIntersectionShader(L"MyIntersectionShader");
	hitGroup.SetHitGroupType(D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE);

	rayTracingPso->Finalize();
	rayTracingPso->Name = Name;

	return rayTracingPso;
}



std::unique_ptr<TopLevelAccelerationStructure> CVGI::ClusterVisibility::BuildAccelerationStructures(ComputeContext& context)
{

	auto& aabbBufferManager = m_data->GetBufferManager(BuildAABBsTechnique::Name);
	UINT32 gridOccupiedCount = *aabbBufferManager.ReadFromBuffer<UINT32*>(context, 4);

	std::vector<ClusterAABBInfo> clusterAABBInfo(gridOccupiedCount);

	GPUBuffer& aabbBuffer = aabbBufferManager.GetBuffer(0);
	// Not flushing because ReadFromBuffer will do it
	context.TransitionResource(aabbBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	UINT32 buffSize = gridOccupiedCount * sizeof(ClusterAABBInfo);

	memcpy(clusterAABBInfo.data(), aabbBufferManager.ReadFromBuffer<ClusterAABBInfo*>(context, 1, buffSize), buffSize);

	std::erase_if(clusterAABBInfo, [](const ClusterAABBInfo& element) {
		return element.ClusterElementCount == 0;
		});

	UINT32 geometriesPerBlas = clusterAABBInfo.size() / MAX_BLAS_COUNT;;
	int remainder = clusterAABBInfo.size() % MAX_BLAS_COUNT;



	std::vector<UINT32> geometryDataBuffer(clusterAABBInfo.size());
	std::vector<UINT32> blasDataBuffer;

	std::unique_ptr<TopLevelAccelerationStructure> tlas = std::make_unique<TopLevelAccelerationStructure>();

	UINT32 idx = 0;
	while (idx < clusterAABBInfo.size())
	{
		std::shared_ptr<BottomLevelAccelerationStructure> blas = std::make_shared<BottomLevelAccelerationStructure>();

		UINT32 numGeometries = remainder > 0 ? geometriesPerBlas + 1 : geometriesPerBlas;

		for (UINT i = 0; i < numGeometries; i++)
		{
			auto& clusterInfo = clusterAABBInfo[idx + i];

			geometryDataBuffer[idx + i] = clusterInfo.ClusterStartIndex;

			blas->AddGeometry(clusterInfo.ClusterElementCount,
				aabbBuffer.GetGpuVirtualAddress() + clusterInfo.ClusterStartIndex * sizeof(AABB),
				sizeof(AABB), AABB(clusterInfo.Min, clusterInfo.Max));
		}

		blas->Build(context);
		tlas->AddBLAS(blas);

		blasDataBuffer.push_back(idx);

		idx += numGeometries;
		remainder--;
	}

	for (UINT32 i = 0; i < clusterAABBInfo.size(); i++)
	{
		assert(geometryDataBuffer[i] == clusterAABBInfo[i].ClusterStartIndex);
	}

	tlas->Build(context);

	UINT blasCount = tlas->GetBLASCount();

	m_bufferManager->ResizeBuffer((UINT)BufferType::GeometryOffset, blasDataBuffer.size());
	m_bufferManager->ResizeBuffer((UINT)BufferType::AABBOffset, clusterAABBInfo.size());

	UploadBuffer geometryDataBufferUpload;
	geometryDataBufferUpload.Create(geometryDataBuffer.size() * sizeof(UINT32));

	UploadBuffer blasDataBufferUpload;
	blasDataBufferUpload.Create(blasDataBuffer.size() * sizeof(UINT32));

	memcpy(blasDataBufferUpload.Map(), blasDataBuffer.data(), blasDataBuffer.size() * sizeof(UINT32));
	memcpy(geometryDataBufferUpload.Map(), geometryDataBuffer.data(), geometryDataBuffer.size() * sizeof(UINT32));


	context.CopyBuffer(m_bufferManager->GetBuffer((UINT)BufferType::GeometryOffset), blasDataBufferUpload);
	context.CopyBuffer(m_bufferManager->GetBuffer((UINT)BufferType::AABBOffset), geometryDataBufferUpload);
	context.Flush(true);

	geometryDataBufferUpload.Unmap();
	blasDataBufferUpload.Unmap();


	context.Flush();
	
	return std::move(tlas);
}

const std::wstring ClusterVisibility::Name = L"ClusterVisibility";
