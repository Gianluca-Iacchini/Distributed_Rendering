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
		Model() : m_minBounds(FLT_MAX, FLT_MAX, FLT_MAX), m_maxBounds(-FLT_MAX, -FLT_MAX, -FLT_MAX) {}
		~Model() {}

		void LoadFromFile(const aiScene* scene);

		void LoadMaterials(const aiScene* scene);

		void LoadMeshes(const aiScene* scene);

		const std::vector<std::shared_ptr<Mesh>>& GetMeshes() const { return m_meshes; }
		std::shared_ptr<Mesh> GetMeshAt(int index) const { return m_meshes[index]; }

		const std::vector<SharedMaterial>& GetMaterials() const { return m_materials; }
		SharedMaterial GetMaterialAt(int index) const { return m_materials[index]; }

		void Draw(ID3D12GraphicsCommandList* commandList);
		void Draw(CommandList* commandList);
		void Draw(CommandContext& context);

		void UseBuffers(CommandContext& context);

		DX12Lib::AABB GetBounds() const { return AABB{ m_minBounds, m_maxBounds }; }

	private:
		void BuildVertexBuffer(UINT stride, UINT numVertices);
		void BuildIndexBuffer(DXGI_FORMAT format, UINT numIndices);


	protected:
		std::vector<std::shared_ptr<Mesh>> m_meshes;
		std::vector<SharedMaterial> m_materials;

		Microsoft::WRL::ComPtr<ID3D12Resource> m_vertexBufferResource;
		Microsoft::WRL::ComPtr<ID3D12Resource> m_indexBufferResource;

		D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
		D3D12_INDEX_BUFFER_VIEW m_indexBufferView;

		DirectX::XMFLOAT3 m_minBounds;
		DirectX::XMFLOAT3 m_maxBounds;
	};
}

