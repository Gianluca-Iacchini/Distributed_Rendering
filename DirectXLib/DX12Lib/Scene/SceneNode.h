#pragma once
#include "Transform.h"
#include "Component.h"
#include "DX12Lib/Models/ModelRenderer.h"


namespace DX12Lib
{
	class SceneNode;

	using NodePtr = std::unique_ptr<SceneNode>;

	class SceneNode
	{
	public:
		SceneNode() : m_dirtyFlags(0) 
		{
			ThrowIfFailed(CoCreateGuid(&m_guid));
		}
		~SceneNode();

		void Update(CommandContext* context);
		void Render(CommandContext* context);

		void AddChild(SceneNode* node);
		SceneNode* AddChild();
		void RemoveChild(SceneNode* node);
		SceneNode* GetChild(SceneNode* node);

		template<typename T, typename... Args>
		inline T* AddComponent(Args&&... args)
		{
			static_assert(std::is_base_of<Component, T>::value, "T must derive from Component");

			std::unique_ptr<T> componentPtr = std::make_unique<T>(std::forward<Args>(args)...);
			componentPtr->Node = this;
			T* returnPtr = componentPtr.get();

			this->m_components.push_back(std::move(componentPtr));

			returnPtr->Init();

			return returnPtr;
		}

		void SetPosition(const DirectX::XMFLOAT3& position);
		void SetPosition(float x, float y, float z);
		void SetRotationQuaternion(const DirectX::XMFLOAT4& rotationQuat);
		void SetRotationEulerAngles(const DirectX::XMFLOAT3& rotationEuler);
		void SetRotationEulerAngles(float pitch, float yaw, float roll);
		void SetScale(const DirectX::XMFLOAT3& scale);
		void SetScale(float x, float y, float z);

		void SetRelativePosition(const DirectX::XMFLOAT3& position);
		void SetRelativePosition(float x, float y, float z);
		void SetRelativeRotation(const DirectX::XMFLOAT4& rotationQuat);
		void SetRelativeRotation(const DirectX::XMFLOAT3& rotationEuler);
		void SetRelativeRotation(float pitch, float yaw, float roll);
		void SetRelativeScale(const DirectX::XMFLOAT3& scale);
		void SetRelativeScale(float x, float y, float z);

		DirectX::XMFLOAT3 GetPosition();
		DirectX::XMFLOAT3 GetRelativePosition();
		DirectX::XMFLOAT4 GetRotationQuaternion();
		DirectX::XMFLOAT4 GetRelativeRotationQuaternion();
		DirectX::XMFLOAT3 GetRotationEulerAngles();
		DirectX::XMFLOAT3 GetRelativeRotationEulerAngles();
		DirectX::XMFLOAT3 GetScale();
		DirectX::XMFLOAT3 GetRelativeScale();

		DirectX::XMFLOAT4X4 GetWorldMatrix4x4();
		DirectX::XMMATRIX GetWorldMatrix();
		DirectX::XMFLOAT4X4 GetWorldInverse4x4();
		DirectX::XMMATRIX GetWorldInverse();

		SceneNode* GetChildAt(UINT index) const 
		{ 
			assert(index < m_children.size());

			return m_children[index].get(); 
		}

	public:
		std::wstring Name = L"";

	public:
		bool operator==(const SceneNode& other) const { return m_guid == other.m_guid; }

	private:
		void PropagateDirtyTransform(UINT dirtyFlag = 5);

	private:
		Transform m_transform;
		void SetParent(SceneNode* parent);

		// Raw pointer because child nodes do not own their parent
		SceneNode* m_parent = nullptr;
		std::vector<NodePtr> m_children;

		UINT m_dirtyFlags;

		enum class ParentDirtyFlags
		{
			Position = 1 << 0,
			Rotation = 1 << 1,
			Scale = 1 << 2
		};


		std::vector<std::unique_ptr<Component>> m_components;


	private:
		GUID m_guid = GUID_NULL;

		
	};




}

