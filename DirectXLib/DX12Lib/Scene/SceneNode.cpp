#include "DX12Lib/pch.h"
#include "SceneNode.h"

using namespace DX12Lib;

DX12Lib::SceneNode::~SceneNode()
{
	// Manually release the vector so the children are deleted before the parent
	// This is important because the the children may refer the parent components
	m_children.clear();
	m_components.clear();
}

void DX12Lib::SceneNode::Update(CommandContext* context)
{
	// Will only update if the transform is dirty
	m_transform.Update();

	if (context != nullptr)
	{
		for (std::unique_ptr<Component>& component : m_components)
		{
			component->Context = context;
			component->Update();
			component->Context = nullptr;
		}
	}

	for (auto& node : m_children) 
		node->Update(context);


}

void DX12Lib::SceneNode::Render(CommandContext* context)
{
	if (context == nullptr)
		return;

	for (std::unique_ptr<Component>& component : m_components)
	{
		component->Context = context;
		component->Render();
		component->Context = nullptr;
	}


	for (auto& node : m_children)
		node->Render(context);
	
}

void DX12Lib::SceneNode::AddChild(SceneNode* node)
{
	assert(node != nullptr);

	if (node->m_parent != nullptr)
	{
		auto& children = node->m_parent->m_children;

		auto iter = std::find_if(children.begin(), children.end(),
			[&node](const std::unique_ptr<SceneNode>& child)
			{
				return *child == *node;
			});

		if (iter != children.end())
		{
			this->m_children.push_back(std::move(*iter));
			children.erase(iter);
		}
	}

	node->SetParent(this);
}

SceneNode* DX12Lib::SceneNode::AddChild()
{
	std::unique_ptr<SceneNode> node = std::make_unique<SceneNode>();
	SceneNode* nodePtr = node.get();

	m_children.push_back(std::move(node));

	nodePtr->SetParent(this);

	return nodePtr;
}



void DX12Lib::SceneNode::RemoveChild(SceneNode* node)
{
	auto iter = std::find_if(m_children.begin(), m_children.end(),
		[node](const std::unique_ptr<DX12Lib::SceneNode>& child) {
			return child.get() == node;
		});

	if (iter != m_children.end())
	{
		m_children.erase(iter);
	}
}

SceneNode* DX12Lib::SceneNode::GetChild(SceneNode* node)
{
	auto iter = std::find_if(m_children.begin(), m_children.end(),
		[node](const std::unique_ptr<DX12Lib::SceneNode>& child) {
			return child.get() == node;
		});

	if (iter != m_children.end())
	{
		return iter->get();
	}

	return nullptr;
}

void DX12Lib::SceneNode::PropagateDirtyTransform(UINT dirtyFlag)
{
	m_transform.SetDirty(dirtyFlag);

	for (auto& node : m_children)
	{
		node->PropagateDirtyTransform(dirtyFlag);
	}
}

void SceneNode::SetParent(SceneNode* parent)
{
	this->m_parent = parent;

	if (parent != nullptr)
	{
		m_transform.m_parent = &parent->m_transform;
	}

	PropagateDirtyTransform();
}


void DX12Lib::SceneNode::SetPosition(const DirectX::XMFLOAT3& position)
{
	m_transform.SetWorldPosition(position);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Position);
}

void DX12Lib::SceneNode::SetPosition(float x, float y, float z)
{
	this->SetPosition(DirectX::XMFLOAT3(x, y, z));
}

void DX12Lib::SceneNode::SetRotationQuaternion(const DirectX::XMFLOAT4& quatRotation)
{
	m_transform.SetWorldRotation(quatRotation);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Rotation);
}

void DX12Lib::SceneNode::SetRotationEulerAngles(const DirectX::XMFLOAT3& rotationEuler)
{
	DirectX::XMVECTOR rotation = DirectX::XMQuaternionRotationRollPitchYaw(rotationEuler.x, rotationEuler.y, rotationEuler.z);
	m_transform.SetWorldRotation(rotation);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Rotation);

}

void DX12Lib::SceneNode::SetRotationEulerAngles(float pitch, float yaw, float roll)
{
	this->SetRotationEulerAngles(DirectX::XMFLOAT3(pitch, yaw, roll));
}

void DX12Lib::SceneNode::SetScale(const DirectX::XMFLOAT3& scale)
{
	m_transform.SetWorldScale(scale);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Scale);
}

void DX12Lib::SceneNode::SetScale(float x, float y, float z)
{
	this->SetScale(DirectX::XMFLOAT3(x, y, z));
}

void DX12Lib::SceneNode::SetRelativePosition(const DirectX::XMFLOAT3& position)
{
	m_transform.SetRelativePosition(position);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Position);
}

void DX12Lib::SceneNode::SetRelativePosition(float x, float y, float z)
{
	this->SetRelativePosition(DirectX::XMFLOAT3(x, y, z));
}

void DX12Lib::SceneNode::SetRelativeRotation(const DirectX::XMFLOAT4& quatRotation)
{
	m_transform.SetRelativeRotation(quatRotation);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Rotation);
}

void DX12Lib::SceneNode::SetRelativeRotation(const DirectX::XMFLOAT3& rotationEuler)
{
	DirectX::XMVECTOR rotation = DirectX::XMQuaternionRotationRollPitchYaw(rotationEuler.x, rotationEuler.y, rotationEuler.z);
	m_transform.SetRelativeRotation(rotation);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Rotation);
}

void DX12Lib::SceneNode::SetRelativeRotation(float pitch, float yaw, float roll)
{
	this->SetRelativeRotation(DirectX::XMFLOAT3(pitch, yaw, roll));
}

void DX12Lib::SceneNode::SetRelativeScale(const DirectX::XMFLOAT3& scale)
{
	m_transform.SetRelativeScale(scale);
	PropagateDirtyTransform((UINT)Transform::DirtyFlags::Scale);
}

void DX12Lib::SceneNode::SetRelativeScale(float x, float y, float z)
{
	this->SetRelativeScale(DirectX::XMFLOAT3(x, y, z));
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetPosition()
{
	return m_transform.GetWorldPosition3f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetRelativePosition()
{
	return m_transform.GetRelativePosition3f();
}

DirectX::XMFLOAT4 DX12Lib::SceneNode::GetRotationQuaternion()
{
	return m_transform.GetWorldRotation4f();
}

DirectX::XMFLOAT4 DX12Lib::SceneNode::GetRelativeRotationQuaternion()
{
	return m_transform.GetRelativeRotation4f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetRotationEulerAngles()
{
	return m_transform.GetWorldRotationEuler3f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetRelativeRotationEulerAngles()
{
	return m_transform.GetRelativeRotationEuler3f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetScale()
{
	return m_transform.GetWorldScale3f();
}

DirectX::XMFLOAT3 DX12Lib::SceneNode::GetRelativeScale()
{
	return m_transform.GetRelativeScale3f();
}

DirectX::XMFLOAT4X4 DX12Lib::SceneNode::GetWorldMatrix4x4()
{
	DirectX::XMFLOAT4X4 world;
	DirectX::XMStoreFloat4x4(&world, m_transform.GetWorld());

	return world;
}


DirectX::XMFLOAT4X4 DX12Lib::SceneNode::GetWorldInverse4x4()
{
	DirectX::XMFLOAT4X4 world4x4;
	DirectX::XMMATRIX world = DirectX::XMMatrixInverse(nullptr, m_transform.GetWorld());
	DirectX::XMStoreFloat4x4(&world4x4, world);

	return world4x4;
}

DirectX::XMMATRIX DX12Lib::SceneNode::GetWorldMatrix()
{
	return m_transform.GetWorld();
}

DirectX::XMMATRIX DX12Lib::SceneNode::GetWorldInverse()
{
	return DirectX::XMMatrixInverse(nullptr, m_transform.GetWorld());
}

