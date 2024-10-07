#pragma once

#include "DX12Lib/DXWrapper/PipelineState.h"
#include "DX12Lib/DXWrapper/UploadBuffer.h"
#include "d3dx12.h"

namespace CVGI
{
	class ShaderRecord
	{
	public:
		ShaderRecord(void* shaderIdentifier, UINT shaderIdentifierSize, void* localRootArguments, UINT localRootArgumentsSize) :
			m_shaderIdentifier(shaderIdentifier, shaderIdentifierSize), m_localRootArgs(localRootArguments, localRootArgumentsSize) {}

		ShaderRecord(void* shaderIdentifier, UINT shaderIdentifierSize) :
			m_shaderIdentifier(shaderIdentifier, shaderIdentifierSize) {}

		void CopyTo(void* dest) const
		{
			uint8_t* byteDest = static_cast<uint8_t*>(dest);
			memcpy(byteDest, m_shaderIdentifier.ptr, m_shaderIdentifier.size);
			if (m_localRootArgs.ptr)
			{
				memcpy(byteDest + m_shaderIdentifier.size, m_localRootArgs.ptr, m_localRootArgs.size);
			}
		}

		struct PointerWithSize {
			void* ptr;
			UINT size;

			PointerWithSize() : ptr(nullptr), size(0) {}
			PointerWithSize(void* _ptr, UINT _size) : ptr(_ptr), size(_size) {};
		};
		PointerWithSize m_shaderIdentifier;
		PointerWithSize m_localRootArgs;
	};

	class ShaderTable : public DX12Lib::UploadBuffer
	{
	private:
		uint8_t* m_mappedShaderRecords = nullptr;
		UINT m_shaderRecordSize = 0;
		std::vector<ShaderRecord> m_shaderRecords;

	public:
		ShaderTable() = default;

		ShaderTable(UINT numShaderRecords, UINT shaderRecordSize)
		{
			m_shaderRecordSize = (shaderRecordSize + D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1) & ~(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT - 1);
			m_shaderRecords.reserve(numShaderRecords);

			UINT buffSize = numShaderRecords * shaderRecordSize;

			this->Create(buffSize);
			m_mappedShaderRecords = static_cast<uint8_t*>(this->Map());
		}

		void PushRecord(const ShaderRecord& shaderRecord)
		{
			assert(m_shaderRecords.size() < m_shaderRecords.capacity());
			m_shaderRecords.push_back(shaderRecord);
			shaderRecord.CopyTo(m_mappedShaderRecords);
			m_mappedShaderRecords += m_shaderRecordSize;
		}

		UINT GetShaderRecordSize() const { return m_shaderRecordSize; }

		virtual ~ShaderTable()
		{
			if (m_isMapped)
				this->Unmap();
		}
	};

	enum class RayTracingShaderType
	{
		Raygen = 0,
		ClosestHit,
		AnyHit,
		Miss,
		Intersection,
		Count
	};

	class HitGroup
	{
	public:
		HitGroup(std::wstring hitGroupName, CD3DX12_HIT_GROUP_SUBOBJECT*& hitGroup) : name(hitGroupName)
		{
			hitGroupSubObject = hitGroup;
			hitGroupSubObject->SetHitGroupExport(name.c_str());
		}
		~HitGroup() {}
		
		void AddClosestHitShader(std::wstring shaderName) { hitGroupSubObject->SetClosestHitShaderImport(shaderName.c_str()); }
		void AddAnyHitShader(std::wstring shaderName) { hitGroupSubObject->SetAnyHitShaderImport(shaderName.c_str()); }
		void AddIntersectionShader(std::wstring shaderName) { hitGroupSubObject->SetIntersectionShaderImport(shaderName.c_str()); }
		void SetHitGroupType(D3D12_HIT_GROUP_TYPE type) { hitGroupSubObject->SetHitGroupType(type); }

	private:
		const std::wstring name;
		CD3DX12_HIT_GROUP_SUBOBJECT* hitGroupSubObject;
	};

	class RaytracingStateObject : public DX12Lib::PipelineState
	{
	public:


	public:
		RaytracingStateObject() : m_raytracingPipeline(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE) {}
		virtual ~RaytracingStateObject() {}
		void Finalize();

		virtual void Use(DX12Lib::CommandList& commandList) const override;
		virtual void UseRootSignature(DX12Lib::CommandList& commandList) const override;

		D3D12_DISPATCH_RAYS_DESC GetDefaultDispatchDesc() const { return m_defaultDispatchDesc; }

		void SetRecursionDepth(UINT maxRecursionDepth = 1);
		void SetAttributeAndPayloadSize(UINT attributeSize, UINT payloadSize);
		void SetLocalRootSignature(std::wstring exportName, std::shared_ptr<DX12Lib::RootSignature> rootSignature);



		HitGroup& CreateHitGroup(std::wstring hitGroupName = L"");

		void SetShaderBytecode(D3D12_SHADER_BYTECODE shaderBytecode);
		void SetShaderEntryPoint(RayTracingShaderType shaderType, std::wstring entryPoint);


	private:
		void CreateShaderTables();
	private:

		D3D12_DISPATCH_RAYS_DESC m_defaultDispatchDesc = {};

		std::unique_ptr<ShaderTable> m_missShaderTable;
		std::unique_ptr<ShaderTable> m_hitGroupShaderTable;
		std::unique_ptr<ShaderTable> m_rayGenShaderTable;

		Microsoft::WRL::ComPtr<ID3D12StateObject> m_stateObject;
		CD3DX12_STATE_OBJECT_DESC m_raytracingPipeline;



		std::unordered_map<std::wstring, HitGroup> m_hitGroups;

		CD3DX12_DXIL_LIBRARY_SUBOBJECT* m_shaderLib = nullptr;

		std::vector<std::wstring> v_shaderNames [(UINT)RayTracingShaderType::Count];
	};
}