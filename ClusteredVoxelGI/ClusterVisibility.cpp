#include "ClusterVisibility.h"
#include "DX12Lib/pch.h"
#include "WinPixEventRuntime/pix3.h"
#include "DX12Lib/Scene/SceneCamera.h"
#include "dxcapi.h"
#include "Raytracing.h"

#define MAX_BLAS_COUNT 250

using namespace CVGI;
using namespace DX12Lib;
using namespace Graphics;


void CVGI::ClusterVisibility::InitializeBuffers(UINT voxelCount, UINT clusterCount)
{
	// Faces buffer (u0)
	m_faceBufferManager.AddStructuredBuffer(1, sizeof(DirectX::XMUINT2));
	// Face start count buffer (u1)
	m_faceBufferManager.AddStructuredBuffer(voxelCount, sizeof(DirectX::XMUINT2));
	// Number of faces (u2);
	m_faceBufferManager.AddByteAddressBuffer();

	// Buffers for acceleration structures
	// Voxel AABB data (u0);
	m_aabbBufferManager.AddStructuredBuffer(voxelCount, sizeof(AABB));
	// Cluster Start Index and Count (u1);
	DirectX::XMUINT3 groupSize = MathHelper::Ceil(m_voxelTexDimensions, 8);
	m_aabbBufferManager.AddStructuredBuffer(groupSize.x * groupSize.y * groupSize.z, sizeof(AABBInfo));
	// AABB voxel indices (u2);
	m_aabbBufferManager.AddStructuredBuffer(voxelCount, sizeof(UINT32));
	// Cluster AABB Counter (u3);
	m_aabbBufferManager.AddByteAddressBuffer();
	// Debug AABB Coutner (u4);
	m_aabbBufferManager.AddByteAddressBuffer();

	m_raytracingBufferManager.AddStructuredBuffer(1, sizeof(UINT32));
	m_raytracingBufferManager.AddStructuredBuffer(1, sizeof(UINT32));
	m_raytracingBufferManager.AddStructuredBuffer(1, sizeof(DirectX::XMUINT2));
	m_raytracingBufferManager.AddStructuredBuffer(1, sizeof(UINT32));
	m_raytracingBufferManager.Add2DTextureBuffer(Renderer::s_clientWidth, Renderer::s_clientHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
	m_raytracingBufferManager.AddByteAddressBuffer();

	m_faceBufferManager.AllocateBuffers();
	m_aabbBufferManager.AllocateBuffers();
	m_raytracingBufferManager.AllocateBuffers();


	m_numberOfVoxels = voxelCount;
	m_numberOfClusters = clusterCount;
}

void CVGI::ClusterVisibility::StartVisibility(ComputeContext& context, BufferManager& compactBufferManager)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"FaceCount");

	m_cbFaceCount.GridDimension = m_voxelTexDimensions;
	m_cbFaceCount.CurrentPhase = 0;
	m_cbFaceCount.VoxelCount = m_numberOfVoxels;

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_faceBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.FlushResourceBarriers();

	VisibilityPass(context, DirectX::XMUINT3(ceil(m_numberOfVoxels / 256), 1, 1), compactBufferManager);
	context.Flush();

	m_cbFaceCount.CurrentPhase = 1;

	m_numberOfFaces = *m_faceBufferManager.ReadFromBuffer<UINT32*>(context, 2);
	m_faceBufferManager.ResizeBuffer(0, m_numberOfFaces);
	m_faceBufferManager.ZeroBuffer(context, 2);

	VisibilityPass(context, DirectX::XMUINT3(ceil(m_numberOfVoxels / 256), 1, 1), compactBufferManager);

	PIXEndEvent(context.m_commandList->Get());

	context.Flush();
}

void CVGI::ClusterVisibility::VisibilityPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[FaceCountPsoName].get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)FaceCountRootSignature::FaceCountCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbFaceCount).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FaceCountRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)FaceCountRootSignature::FaceCountUAVTable, m_faceBufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}

void CVGI::ClusterVisibility::StartAABBGeneration(ComputeContext& context, BufferManager& compactBufferManager, BufferManager& clusterBufferManager)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"AABBGeneration");

	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	m_aabbBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	context.FlushResourceBarriers();

	m_cbAABBGeneration.GridDimension = m_voxelTexDimensions;
	m_cbAABBGeneration.ClusterCount = m_numberOfClusters;

	AABBGenerationPass(context, MathHelper::Ceil(m_voxelTexDimensions, 8), compactBufferManager, clusterBufferManager);
	
	m_gridOccupiedCount = *m_aabbBufferManager.ReadFromBuffer<UINT32*>(context, 4);

	DXLIB_CORE_INFO("Debug Value: {0}", m_gridOccupiedCount);

	PIXEndEvent(context.m_commandList->Get());
	context.Flush();
}

void CVGI::ClusterVisibility::AABBGenerationPass(DX12Lib::ComputeContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager, BufferManager& clusterBufferManager)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[AABBGenerationPsoName].get());

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)AABBGenerationRootSignature::AABBGenerationCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbAABBGeneration).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)AABBGenerationRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)AABBGenerationRootSignature::ClusterSRVTable, clusterBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)AABBGenerationRootSignature::AABBGenerationUAVTable, m_aabbBufferManager.GetUAVHandle());

	context.Dispatch(groupSize.x, groupSize.y, groupSize.z);
}



void CVGI::ClusterVisibility::ClusterRayTrace(BufferManager& compactBufferManager, BufferManager& clusterBufferManager)
{
	RayTracingContext& context = RayTracingContext::Begin();
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 128, 0), L"RayTracing");

	UINT side = floor(std::cbrt(m_numberOfFaces));
	DirectX::XMUINT3 dispatchSize = DirectX::XMUINT3(side+1, side+1, side);

	m_cbRayTracing.CurrentPhase = 0;
	m_cbRayTracing.DispatchSize = dispatchSize;
	m_cbRayTracing.NumberOfFaces = m_numberOfFaces;
	m_cbRayTracing.GridDimension = m_voxelTexDimensions;
	m_cbRayTracing.Rand1 = MathHelper::RandF();
	m_cbRayTracing.Rand2 = MathHelper::RandF();

	assert(dispatchSize.x * dispatchSize.y * dispatchSize.z >= m_numberOfFaces);

	ClusterRayTracePass(context, dispatchSize, compactBufferManager, clusterBufferManager);
	
	UINT32 visibleClusterCount = *m_raytracingBufferManager.ReadFromBuffer<UINT32*>(context, 5);


	m_raytracingBufferManager.ResizeBuffer(3, visibleClusterCount);


	m_cbRayTracing.CurrentPhase = 1;
	ClusterRayTracePass(context, dispatchSize, compactBufferManager, clusterBufferManager);

	PIXEndEvent(context.m_commandList->Get());
	context.Finish(true);


}

void CVGI::ClusterVisibility::ClusterRayTracePass(RayTracingContext& context, DirectX::XMUINT3 groupSize, BufferManager& compactBufferManager, BufferManager& clusterBufferManager)
{
	context.SetDescriptorHeap(Renderer::s_textureHeap.get());
	context.SetPipelineState(Renderer::s_PSOs[L"RayTracingPso"].get());

	m_raytracingBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	m_aabbBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	compactBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	clusterBufferManager.TransitionAll(context, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	context.FlushResourceBarriers();

	context.m_commandList->Get()->SetComputeRootConstantBufferView((UINT)RayTraceRootSignature::RayTraceCBV, Renderer::s_graphicsMemory->AllocateConstant(m_cbRayTracing).GpuAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::CompactSRVTable, compactBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::ClusterSRVTable, clusterBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::FaceSRVTable, m_faceBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::AABBSRVTable, m_aabbBufferManager.GetSRVHandle());
	context.m_commandList->Get()->SetComputeRootShaderResourceView((UINT)RayTraceRootSignature::AccelerationStructureSRV, m_TLAS.GetGpuVirtualAddress());
	context.m_commandList->Get()->SetComputeRootDescriptorTable((UINT)RayTraceRootSignature::RayTraceUAVTable, m_raytracingBufferManager.GetUAVHandle());
	context.DispatchRays3D(groupSize.x, groupSize.y, groupSize.z);
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusterVisibility::BuildFaceCountRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> faceCountRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)FaceCountRootSignature::Count, 1);
	faceCountRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::FaceCountCBV].InitAsConstantBuffer(0);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*faceCountRootSignature)[(UINT)FaceCountRootSignature::FaceCountUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 0);

	faceCountRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return faceCountRootSignature;
}


std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::ClusterVisibility::BuildFaceCountPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\FaceCount_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> faceCountComputePso = std::make_shared<ComputePipelineState>();
	faceCountComputePso->SetRootSignature(rootSig);
	faceCountComputePso->SetComputeShader(computeShader);
	faceCountComputePso->Finalize();
	faceCountComputePso->Name = FaceCountPsoName;

	return faceCountComputePso;
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusterVisibility::BuildAABBGenerationRootSignature()
{
	SamplerDesc defaultSamplerDesc;

	std::shared_ptr<DX12Lib::RootSignature> AABBComputeRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)AABBGenerationRootSignature::Count, 1);
	AABBComputeRootSignature->InitStaticSampler(0, defaultSamplerDesc);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::AABBGenerationCBV].InitAsConstantBuffer(0);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::ClusterSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*AABBComputeRootSignature)[(UINT)AABBGenerationRootSignature::AABBGenerationUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 5, D3D12_SHADER_VISIBILITY_ALL, 0);

	AABBComputeRootSignature->Finalize(D3D12_ROOT_SIGNATURE_FLAG_NONE);

	return AABBComputeRootSignature;
}

std::shared_ptr<DX12Lib::ComputePipelineState> CVGI::ClusterVisibility::BuildAABBGenerationPipelineState(std::shared_ptr<DX12Lib::RootSignature> rootSig)
{
	std::wstring shaderPath = Utils::ToWstring(SOURCE_DIR) + L"\\Shaders";
	std::wstring computeShaderPath = shaderPath + L"\\ComputeAABB_CS.hlsl";

	std::shared_ptr<Shader> computeShader = std::make_shared<Shader>(computeShaderPath, "CS", "cs_5_1");
	computeShader->Compile();

	std::shared_ptr<ComputePipelineState> AABBGenerationComputePso = std::make_shared<ComputePipelineState>();
	AABBGenerationComputePso->SetRootSignature(rootSig);
	AABBGenerationComputePso->SetComputeShader(computeShader);
	AABBGenerationComputePso->Finalize();
	AABBGenerationComputePso->Name = AABBGenerationPsoName;

	return AABBGenerationComputePso;
}

std::shared_ptr<DX12Lib::RootSignature> CVGI::ClusterVisibility::BuildRaytracingGlobalRootSignature()
{
	std::shared_ptr<DX12Lib::RootSignature> rayTracingRootSignature = std::make_shared<DX12Lib::RootSignature>((UINT)RayTraceRootSignature::Count, 0);

	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::RayTraceCBV].InitAsConstantBuffer(0);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::AccelerationStructureSRV].InitAsBufferSRV(0, D3D12_SHADER_VISIBILITY_ALL, 0);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::CompactSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 4, D3D12_SHADER_VISIBILITY_ALL, 1);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::ClusterSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 2);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::FaceSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 1, D3D12_SHADER_VISIBILITY_ALL, 3);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::AABBSRVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 3, D3D12_SHADER_VISIBILITY_ALL, 4);
	(*rayTracingRootSignature)[(UINT)RayTraceRootSignature::RayTraceUAVTable].InitAsDescriptorRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 6);


	rayTracingRootSignature->Finalize();

	return rayTracingRootSignature;
}

std::shared_ptr<CVGI::RaytracingStateObject> CVGI::ClusterVisibility::BuildRayTracingPipelineState(std::shared_ptr<DX12Lib::RootSignature> globalRootSig)
{
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
	rayTracingPso->Name = L"RayTracingPso";

	return rayTracingPso;
}

void CVGI::ClusterVisibility::BuildAccelerationStructures(ComputeContext& context)
{
	PIXBeginEvent(context.m_commandList->Get(), PIX_COLOR(128, 0, 0), L"BuildAccelerationStructures");

	std::vector<AABBInfo> clusterAABBInfo(m_gridOccupiedCount);

	GPUBuffer& aabbBuffer = m_aabbBufferManager.GetBuffer(0);
	// Not flushing because ReadFromBuffer will do it
	context.TransitionResource(aabbBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

	UINT32 buffSize = m_gridOccupiedCount * sizeof(AABBInfo);

	memcpy(clusterAABBInfo.data(), m_aabbBufferManager.ReadFromBuffer<AABBInfo*>(context, 1, buffSize), buffSize);
	
	std::erase_if(clusterAABBInfo, [](const AABBInfo& element) {
		return element.ClusterElementCount == 0;
		});

	UINT32 geometriesPerBlas = clusterAABBInfo.size() / MAX_BLAS_COUNT;;
	int remainder = clusterAABBInfo.size() % MAX_BLAS_COUNT;



	std::vector<UINT32> geometryDataBuffer(clusterAABBInfo.size());
	std::vector<UINT32> blasDataBuffer;

	UINT32 idx = 0;
	while (idx < clusterAABBInfo.size())
	{
		std::shared_ptr<BottomLevelAccelerationStructure> blas = std::make_shared<BottomLevelAccelerationStructure>();

		UINT32 numGeometries = remainder > 0 ? geometriesPerBlas + 1 : geometriesPerBlas;

		for (UINT i = 0; i < numGeometries; i++)
		{
			geometryDataBuffer[idx + i] = clusterAABBInfo[idx + i].ClusterStartIndex;
			blas->AddGeometry(clusterAABBInfo[idx + i].ClusterElementCount,
				aabbBuffer.GetGpuVirtualAddress() + clusterAABBInfo[idx + i].ClusterStartIndex * sizeof(AABB),
				sizeof(AABB));
		}

		blas->Build(context);
		m_TLAS.AddBLAS(blas);

		blasDataBuffer.push_back(idx);

		idx += numGeometries;
		remainder--;
	}

	for (UINT32 i = 0; i < clusterAABBInfo.size(); i++)
	{
		assert(geometryDataBuffer[i] == clusterAABBInfo[i].ClusterStartIndex);
	}

	m_TLAS.Build(context);

	UINT blasCount = m_TLAS.GetBLASCount();

	m_raytracingBufferManager.ResizeBuffer(0, blasDataBuffer.size());
	m_raytracingBufferManager.ResizeBuffer(1, clusterAABBInfo.size());
	m_raytracingBufferManager.ResizeBuffer(2, m_numberOfFaces);

	UploadBuffer geometryDataBufferUpload;
	geometryDataBufferUpload.Create(geometryDataBuffer.size() * sizeof(UINT32));

	UploadBuffer blasDataBufferUpload;
	blasDataBufferUpload.Create(blasDataBuffer.size() * sizeof(UINT32));

	memcpy(blasDataBufferUpload.Map(), blasDataBuffer.data(), blasDataBuffer.size() * sizeof(UINT32));
	memcpy(geometryDataBufferUpload.Map(), geometryDataBuffer.data(), geometryDataBuffer.size() * sizeof(UINT32));


	context.CopyBuffer(m_raytracingBufferManager.GetBuffer(0), blasDataBufferUpload);
	context.CopyBuffer(m_raytracingBufferManager.GetBuffer(1), geometryDataBufferUpload);
	context.Flush(true);

	geometryDataBufferUpload.Unmap();
	blasDataBufferUpload.Unmap();

	m_cbRayTracing.BlasCount = blasCount;
	m_cbRayTracing.GeometryCount = clusterAABBInfo.size();

	PIXEndEvent(context.m_commandList->Get());
	context.Flush();
}
