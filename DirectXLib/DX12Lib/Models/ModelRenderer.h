#pragma once

#include "DX12Lib/Scene/Component.h"
#include "Model.h"
#include "Mesh.h"
#include "DX12Lib/Commons/Helpers.h"

namespace DX12Lib
{
	class ModelRenderer;
	class Test;

	class MeshRenderer : public Component
	{

	public:
		MeshRenderer(ModelRenderer* modelRenderer)
		{
			assert(modelRenderer != nullptr);

			MaterialPSO = L"opaquePSO";
			m_modelRenderer = modelRenderer;
		}

		virtual ~MeshRenderer();

		virtual void Init() override;

		virtual void Update() override;

		virtual void Render() override;

		void DrawMesh(ID3D12GraphicsCommandList* cmdList);

		void SetMesh(std::shared_ptr<Mesh> mesh)
		{
			m_mesh = mesh;
		}

		void SetMaterial(SharedMaterial material)
		{
			assert (material != nullptr);

			m_meshMaterial = material;
			MaterialPSO = material->GetDefaultPSO();
		}

	private:
		DirectX::GraphicsResource GetObjectCB();


	public:
		std::wstring MaterialPSO = L"opaquePSO";
		ConstantBufferObject m_objectCB;

	private:
		ModelRenderer* m_modelRenderer = nullptr;
		std::shared_ptr<Mesh> m_mesh = nullptr;
		SharedMaterial m_meshMaterial = nullptr;


		bool m_isVisible = true;
		bool m_IsInBatch = false;
	};

	class ModelRenderer : public Component
	{
	private:
		// Raw pointer because the owner of the mesh renderer is the node that contains it
		std::unordered_map<std::wstring, std::vector<MeshRenderer*>> m_psoMeshRendererBatch;


	public:

		ModelRenderer() {}
		virtual ~ModelRenderer() {}

		std::shared_ptr<Model> Model = nullptr;

		void AddMeshRendererToBatch(MeshRenderer* meshRenderer)
		{
			assert(meshRenderer != nullptr);

			m_psoMeshRendererBatch[meshRenderer->MaterialPSO].push_back(meshRenderer);
		}

		void RemoveMeshRendererFromBatch(MeshRenderer* meshRenderer)
		{
			assert(meshRenderer != nullptr);

			auto& batch = m_psoMeshRendererBatch[meshRenderer->MaterialPSO];
			batch.erase(std::remove(batch.begin(), batch.end(), meshRenderer), batch.end());
		}


		virtual void Init() override {}

		virtual void Update() override
		{
			
		}

		virtual void Render() override;

		void DrawMeshes(ID3D12GraphicsCommandList* cmdList, std::wstring psoName);
	};

}