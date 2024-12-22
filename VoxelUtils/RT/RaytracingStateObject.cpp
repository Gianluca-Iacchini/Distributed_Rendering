#include "RaytracingStateObject.h"
#include "DX12Lib/pch.h"

using namespace VOX;
using namespace DirectX;

void VOX::RaytracingStateObject::Finalize()
{
	auto globalRootSignature = m_raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
	globalRootSignature->SetRootSignature(m_rootSignature->Get());

	ThrowIfFailed(Graphics::s_device->GetDXR()->CreateStateObject(m_raytracingPipeline, IID_PPV_ARGS(m_stateObject.GetAddressOf())));
	CreateShaderTables();
}

void VOX::RaytracingStateObject::Use(DX12Lib::CommandList& commandList) const
{
	commandList.GetDXR()->SetPipelineState1(m_stateObject.Get());
}

void VOX::RaytracingStateObject::UseRootSignature(DX12Lib::CommandList& commandList) const
{
	commandList.Get()->SetComputeRootSignature(m_rootSignature->Get());
}

void VOX::RaytracingStateObject::SetRecursionDepth(UINT maxRecursionDepth)
{
	auto pipelineConfig = m_raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
	pipelineConfig->Config(maxRecursionDepth); // Max Recursion Depth
}

void VOX::RaytracingStateObject::SetAttributeAndPayloadSize(UINT attributeSize, UINT payloadSize)
{
	auto configObject = m_raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
	configObject->Config(attributeSize, payloadSize);
}

void VOX::RaytracingStateObject::SetLocalRootSignature(std::wstring exportName, std::shared_ptr<DX12Lib::RootSignature> rootSignature)
{
	auto localRootSignature = m_raytracingPipeline.CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
	localRootSignature->SetRootSignature(rootSignature->Get());

	auto rootSigAssociation = m_raytracingPipeline.CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
	rootSigAssociation->SetSubobjectToAssociate(*localRootSignature);
	rootSigAssociation->AddExport(exportName.c_str());
}

HitGroup& RaytracingStateObject::CreateHitGroup(std::wstring hitGroupName)
{
	std::wstring name = hitGroupName;
	if (name.empty())
	{
		name = L"hitGroup_" + std::to_wstring(m_hitGroups.size());
	}

	auto hitGroupSubobject = m_raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();


	auto result = m_hitGroups.insert({ name, HitGroup(name, hitGroupSubobject) });

	assert(result.second && "HitGroup with same name already exists");

	// Return hit group reference from the unorderer map
	return result.first->second;
}

void VOX::RaytracingStateObject::SetShaderBytecode(D3D12_SHADER_BYTECODE shaderBytecode)
{
	m_shaderLib = m_raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
	m_shaderLib->SetDXILLibrary(&shaderBytecode);
}

void VOX::RaytracingStateObject::SetShaderEntryPoint(RayTracingShaderType shaderType, std::wstring entryPoint)
{
	assert(m_shaderLib != nullptr && "Shader library must be set before setting entry point");
	assert((shaderType != RayTracingShaderType::Count && !entryPoint.empty()) && "Wrong shader type or empty name");

	for (UINT i = 0; i < (UINT)RayTracingShaderType::Count; i++)
	{
		auto& shaderNames = v_shaderNames[i];
		auto element = std::find(shaderNames.begin(), shaderNames.end(), entryPoint);

		assert((element == shaderNames.end()) && "Entrypoint with the same name already exists");
	}


	m_shaderLib->DefineExport(entryPoint.c_str());
	v_shaderNames[(UINT)shaderType].push_back(entryPoint);
}

void VOX::RaytracingStateObject::CreateShaderTables()
{
	std::vector<void*> rayGenShaderIdentifiers(v_shaderNames[(UINT)RayTracingShaderType::Raygen].size());
	std::vector<void*> missShaderIdentifiers(v_shaderNames[(UINT)RayTracingShaderType::Miss].size());
	std::vector<void*> hitGroupShaderIdentifiers(m_hitGroups.size());

	auto GetShaderIdentifiers = [&](auto* stateObjectProperties)
	{
		UINT maxSize = max(max(rayGenShaderIdentifiers.size(), missShaderIdentifiers.size()), hitGroupShaderIdentifiers.size());

		std::vector<std::wstring>& rayGenNames = v_shaderNames[(UINT)RayTracingShaderType::Raygen];

		std::transform(rayGenNames.begin(), rayGenNames.end(), rayGenShaderIdentifiers.begin(), [&](std::wstring& name)
			{
				return stateObjectProperties->GetShaderIdentifier(name.c_str());
			});

		std::vector<std::wstring>& missNames = v_shaderNames[(UINT)RayTracingShaderType::Miss];

		std::transform(missNames.begin(), missNames.end(), missShaderIdentifiers.begin(), [&](std::wstring& name)
			{
				return stateObjectProperties->GetShaderIdentifier(name.c_str());
			});


		std::transform(m_hitGroups.begin(), m_hitGroups.end(), hitGroupShaderIdentifiers.begin(), [&](auto& pair)
			{
				return stateObjectProperties->GetShaderIdentifier(pair.first.c_str());
			});
	};

	// Get shader identifiers.
	UINT shaderIdentifierSize;
	{
		Microsoft::WRL::ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
		ThrowIfFailed(m_stateObject.As(&stateObjectProperties));
		GetShaderIdentifiers(stateObjectProperties.Get());
		shaderIdentifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
	}

	// Ray gen shader table
	{
		UINT numShaderRecords = v_shaderNames[(UINT)RayTracingShaderType::Raygen].size();
		// Size should be set as the size of the biggest shader identifier, since we are not using local root signature for now,
		// all the shader identifiers will have the same size.
		UINT shaderRecordSize = shaderIdentifierSize;

		m_rayGenShaderTable = std::make_unique<ShaderTable>(numShaderRecords, shaderRecordSize);

		for (auto& rayGenId : rayGenShaderIdentifiers)
		{
			m_rayGenShaderTable->PushRecord(ShaderRecord(rayGenId, shaderIdentifierSize));
		}
	}

	// Miss shader table
	{
		UINT numShaderRecords = v_shaderNames[(UINT)RayTracingShaderType::Miss].size();
		UINT shaderRecordSize = shaderIdentifierSize;

		m_missShaderTable = std::make_unique<ShaderTable>(numShaderRecords, shaderRecordSize);

		for (auto& missId : missShaderIdentifiers)
		{
			m_missShaderTable->PushRecord(ShaderRecord(missId, shaderIdentifierSize));
		}
	}

	// Hit group shader table
	{

		UINT numShaderRecords = m_hitGroups.size();
		UINT shaderRecordSize = shaderIdentifierSize;

		m_hitGroupShaderTable = std::make_unique<ShaderTable>(numShaderRecords, shaderRecordSize);

		for (auto& hitGroupId : hitGroupShaderIdentifiers)
		{
			m_hitGroupShaderTable->PushRecord(ShaderRecord(hitGroupId, shaderIdentifierSize));
		}
	}

	m_defaultDispatchDesc = {};
	m_defaultDispatchDesc.HitGroupTable.StartAddress = m_hitGroupShaderTable->GetGpuVirtualAddress();
	m_defaultDispatchDesc.HitGroupTable.SizeInBytes = m_hitGroupShaderTable->GetDesc().Width;
	m_defaultDispatchDesc.HitGroupTable.StrideInBytes = m_hitGroupShaderTable->GetShaderRecordSize();

	m_defaultDispatchDesc.MissShaderTable.StartAddress = m_missShaderTable->GetGpuVirtualAddress();
	m_defaultDispatchDesc.MissShaderTable.SizeInBytes = m_missShaderTable->GetDesc().Width;
	m_defaultDispatchDesc.MissShaderTable.StrideInBytes = m_missShaderTable->GetShaderRecordSize(); 

	m_defaultDispatchDesc.RayGenerationShaderRecord.StartAddress = m_rayGenShaderTable->GetGpuVirtualAddress();
	m_defaultDispatchDesc.RayGenerationShaderRecord.SizeInBytes = m_rayGenShaderTable->GetShaderRecordSize();

	m_defaultDispatchDesc.Width = Graphics::Renderer::s_clientWidth;
	m_defaultDispatchDesc.Height = Graphics::Renderer::s_clientHeight;
	m_defaultDispatchDesc.Depth = 1;
}
