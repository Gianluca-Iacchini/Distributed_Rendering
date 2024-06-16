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

			MaterialPSO = PSO_PHONG_OPAQUE;
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
		}

		void SetMaterial(SharedMaterial material)
		{
			assert (material != nullptr);

			m_meshMaterial = material;
			MaterialPSO = material->GetDefaultPSO();
		}
		
		Material* GetMaterial() const
		{
			return m_meshMaterial.get();
		}

		const Mesh* GetMesh() const
		{
			return m_mesh.get();
		}

	private:
		DirectX::GraphicsResource GetObjectCB();


	public:
		std::wstring MaterialPSO = PSO_PHONG_OPAQUE;
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
		std::vector<MeshRenderer*> m_opaqueMeshes;
		std::vector<MeshRenderer*> m_transparentMeshes;

	public:

		ModelRenderer() {}
		virtual ~ModelRenderer() {}

		std::shared_ptr<Model> Model = nullptr;

		void AddMeshRendererToBatch(MeshRenderer* meshRenderer)
		{
			assert(meshRenderer != nullptr);

			m_psoMeshRendererBatch[meshRenderer->MaterialPSO].push_back(meshRenderer);
			
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

			auto& batch = m_psoMeshRendererBatch[meshRenderer->MaterialPSO];
			batch.erase(std::remove(batch.begin(), batch.end(), meshRenderer), batch.end());
			m_opaqueMeshes.erase(std::remove(m_opaqueMeshes.begin(), m_opaqueMeshes.end(), meshRenderer), m_opaqueMeshes.end());
			m_transparentMeshes.erase(std::remove(m_transparentMeshes.begin(), m_transparentMeshes.end(), meshRenderer), m_transparentMeshes.end());
		}


		virtual void Init(CommandContext& context) override {}

		virtual void Render(CommandContext& context) override;

		
		void DrawAll(CommandContext& context);
		void DrawOpaque(CommandContext& context);
		void DrawTransparent(CommandContext& context);
		void DrawAllBatch(CommandContext& context, std::wstring psoName);
		void DrawBatchOpaque(CommandContext& context, std::wstring psoName);
		void DrawBatchTransparent(CommandContext& context, std::wstring psoName);

	private:
		void DrawMeshes(CommandContext& context, std::vector<MeshRenderer*> meshRenderers);

	};

}