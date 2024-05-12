#pragma once
#include "assimp/scene.h"
#include "DX12Lib/DXWrapper/DescriptorHeap.h"
#include "MaterialManager.h"
#include "d3d12.h"
#include <wrl/client.h>

namespace DX12Lib {

	class Mesh;
	class CommandContext;
	class CommandList;

	class Model
	{
	public:
		Model() {}
		~Model() {}

		bool LoadFromFile(const std::wstring filename);
		bool LoadFromFile(const char* filename);

		void LoadMaterials(const aiScene* scene);

		void LoadMeshes(const aiScene* scene);

		std::vector<std::shared_ptr<Mesh>> GetMeshes() const { return m_meshes; }

		void Draw(ID3D12GraphicsCommandList* commandList);
		void Draw(CommandList& commandList);
		void Draw(CommandContext& context);


	private:
		void BuildVertexBuffer(UINT stride, UINT numVertices);
		void BuildIndexBuffer(DXGI_FORMAT format, UINT numIndices);

		void BuildMaterialStructuredBuffers();

	protected:
		std::vector<std::shared_ptr<Mesh>> m_meshes;
		std::vector<SharedMaterial> m_materials;

		std::unordered_map<int, int> m_materialIndexMap;
		UINT m_PBRMaterialCount = 0;
		UINT m_phongMaterialCount = 0;

		Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBufferResource;
		Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBufferResource;

		Microsoft::WRL::ComPtr<ID3D12Resource> m_phongMaterialBuffer;
		Microsoft::WRL::ComPtr<ID3D12Resource> m_pbrMaterialBuffer;

		D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
		D3D12_INDEX_BUFFER_VIEW m_indexBufferView;
	};
}

