#include "DX12Lib/pch.h"
#include "ModelRenderer.h"
#include "DX12Lib/Scene/SceneNode.h"

DX12Lib::MeshRenderer::~MeshRenderer()
{
	if (m_IsInBatch)
	{
		if (m_modelRenderer != nullptr)
			m_modelRenderer->RemoveMeshRendererFromBatch(this);
	}
}

void DX12Lib::MeshRenderer::Init()
{
	m_objectCB = ConstantBufferObject();
	DirectX::XMStoreFloat4x4(&m_objectCB.world, DirectX::XMMatrixTranspose(Node->GetWorldMatrix()));
	DirectX::XMStoreFloat4x4(&m_objectCB.invWorld, DirectX::XMMatrixTranspose(Node->GetWorldInverse()));
	m_objectCB.texTransform = MathHelper::Identity4x4();
	m_objectCB.materialIndex = 0;
}

void DX12Lib::MeshRenderer::Update()
{
	// If is visible (not occluded, inside frustum, etc)
	m_isVisible = true;
}

void DX12Lib::MeshRenderer::Render()
{

	if (m_modelRenderer == nullptr)
		return;

	if (m_isVisible)
	{
		if (!m_IsInBatch)
		{
			m_modelRenderer->AddMeshRendererToBatch(this);
			m_IsInBatch = true;
		}
	}

	else if (!m_isVisible)
	{
		if (m_IsInBatch)
		{
			m_modelRenderer->RemoveMeshRendererFromBatch(this);
			m_IsInBatch = false;
		}
	}

	if (m_mesh == nullptr || m_meshMaterial == nullptr)
		return;

}

void DX12Lib::MeshRenderer::DrawMesh(CommandContext* context)
{
	assert(context != nullptr && "Context is null");

	if (m_mesh == nullptr || m_meshMaterial == nullptr)
		return;

	context->m_commandList->Get()->SetGraphicsRootConstantBufferView(1, GetObjectCB().GpuAddress());
	m_meshMaterial->UseMaterial(context->m_commandList->Get());
	m_mesh->Draw(context->m_commandList->Get());
}

DirectX::GraphicsResource DX12Lib::MeshRenderer::GetObjectCB()
{

	DirectX::XMStoreFloat4x4(&m_objectCB.world, DirectX::XMMatrixTranspose(Node->GetWorldMatrix()));
	DirectX::XMStoreFloat4x4(&m_objectCB.invWorld, DirectX::XMMatrixTranspose(Node->GetWorldInverse()));
	m_objectCB.texTransform = MathHelper::Identity4x4();
	m_objectCB.materialIndex = m_mesh->m_materialIndex;
	return Graphics::Renderer::s_graphicsMemory->AllocateConstant(m_objectCB);
}

void DX12Lib::ModelRenderer::Render()
{
	if (Model != nullptr)
	{
		Graphics::Renderer::AddRendererToQueue(this);
	}
}

void DX12Lib::ModelRenderer::Draw(CommandContext* context, std::wstring psoName)
{
	assert(context != nullptr && "Context is null");

	if (Model == nullptr)
		return;

	auto batch = m_psoMeshRendererBatch.find(psoName);

	if (batch == m_psoMeshRendererBatch.end())
		return;

	this->Model->UseBuffers(context);

	for (auto& mesh : batch->second)
	{
		mesh->DrawMesh(context);
	}
}
