#pragma once

#include "DX12Lib/Scene/Component.h"
#include "Model.h"
#include "Mesh.h"
#include "DX12Lib/Commons/Helpers.h"
#include "DX12Lib/Commons/Renderer.h"

namespace DX12Lib
{
	class ModelRenderer;
	class Test;
	class CommandContext;

	class MeshRenderer : public Component
	{

	public:
		MeshRenderer(ModelRenderer* modelRenderer)
		{
			assert(modelRenderer != nullptr);

			m_modelRenderer = modelRenderer;
		}

		virtual ~MeshRenderer();

		virtual void Init(CommandContext& context) override;

		virtual void Update(CommandContext& context) override;

		virtual void Render(CommandContext& context) override;

		void DrawMesh(CommandContext& context);

		void SetMesh(std::shared_ptr<Mesh> mesh)
		{
			m_mesh = mesh;

			if (m_meshMaterial == nullptr)
			{
				m_meshMaterial = m_mesh->MeshMaterial;
			}
		}

		void SetMaterial(Material* material);
		
		Material* GetMaterial() const
		{
			return m_meshMaterial;
		}

		const Mesh* GetMesh() const
		{
			return m_mesh.get();
		}

		DirectX::GraphicsResource GetObjectCB();
		const DescriptorHandle& GetMaterialTextureSRV();
	private:



	public:
		ConstantBufferObject m_objectCB;

	private:
		ModelRenderer* m_modelRenderer = nullptr;
		std::shared_ptr<Mesh> m_mesh = nullptr;
		Material* m_meshMaterial = nullptr;


		bool m_isVisible = true;
		bool m_IsInBatch = false;
	};

	class ModelRenderer : public Component
	{
	private:
		// Raw pointer because the owner of the mesh renderer is the node that contains it
		std::unordered_map<std::wstring, std::vector<MeshRenderer*>> m_psoMeshRendererBatch;
		std::vector<MeshRenderer*> m_opaqueMeshes;
		std::vector<MeshRenderer*> m_transparentMeshes;

	public:

		ModelRenderer() {}
		virtual ~ModelRenderer() {}

		std::shared_ptr<Model> Model = nullptr;

		void UpdateMeshRendererBatch(MeshRenderer* meshRenderer)
		{
			assert(meshRenderer != nullptr);

			RemoveMeshRendererFromBatch(meshRenderer);
			AddMeshRendererToBatch(meshRenderer);
		}

		void AddMeshRendererToBatch(MeshRenderer* meshRenderer)
		{
			assert(meshRenderer != nullptr);

			m_psoMeshRendererBatch[meshRenderer->GetMaterial()->GetMaterialPSO()->Name].push_back(meshRenderer);
			
			bool isTransparent = false;
			
			if (meshRenderer->GetMaterial() != nullptr)
				isTransparent = meshRenderer->GetMaterial()->IsTransparent();

			if (isTransparent)
				m_transparentMeshes.push_back(meshRenderer);
			else
				m_opaqueMeshes.push_back(meshRenderer);
		}

		void RemoveMeshRendererFromBatch(MeshRenderer* meshRenderer)
		{
			assert(meshRenderer != nullptr);
			
			for (auto& batch : m_psoMeshRendererBatch)
			{
				batch.second.erase(std::remove(batch.second.begin(), batch.second.end(), meshRenderer), batch.second.end());
			}

			m_opaqueMeshes.erase(std::remove(m_opaqueMeshes.begin(), m_opaqueMeshes.end(), meshRenderer), m_opaqueMeshes.end());
			m_transparentMeshes.erase(std::remove(m_transparentMeshes.begin(), m_transparentMeshes.end(), meshRenderer), m_transparentMeshes.end());
		}


		virtual void Init(CommandContext& context) override {}

		virtual void Render(CommandContext& context) override;

		
		std::vector<DX12Lib::MeshRenderer*> GetAll();
		std::vector<DX12Lib::MeshRenderer*> GetAllOpaque();
		std::vector<DX12Lib::MeshRenderer*> GetAllTransparent();
		std::vector<DX12Lib::MeshRenderer*> GetAllForPSO(std::wstring psoName);
		std::vector<DX12Lib::MeshRenderer*> GetAllOpaqueForPSO(std::wstring psoName);
		std::vector<DX12Lib::MeshRenderer*> GetAllTransparentForPSO(std::wstring psoName);


	};

}