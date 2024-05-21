#pragma once

#include "DX12Lib/Scene/Component.h"
#include "Model.h"
#include "Mesh.h"
#include "DX12Lib/Commons/Helpers.h"

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

		void DrawMesh(CommandContext* context);

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


		virtual void Init(CommandContext& context) override {}

		virtual void Update(CommandContext& context) override
		{
			
		}

		virtual void Render(CommandContext& context) override;

		void Draw(CommandContext* context, std::wstring psoName);
	};

}