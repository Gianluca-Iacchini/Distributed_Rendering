#pragma once

#include <DirectXMath.h>
#include <string>
#include <tgmath.h>

namespace DX12Lib
{
	struct AABB
	{
		DirectX::XMFLOAT3 Min = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
		DirectX::XMFLOAT3 Max = DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f);
	};
}

class MathHelper
{
public:
	static DirectX::XMFLOAT4X4 Identity4x4()
	{
		static DirectX::XMFLOAT4X4 I(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);

		return I;
	}

	template <typename T>
	static T Max(const T& a, const T& b)
	{
		return a > b ? a : b;
	}

	template <typename T>
	static T Min(const T& a, const T& b)
	{
		return a < b ? a : b;
	}

	static DirectX::XMFLOAT3 Min(const DirectX::XMFLOAT3& a, const DirectX::XMFLOAT3& b)
	{
		return DirectX::XMFLOAT3(
			Min(a.x, b.x),
			Min(a.y, b.y),
			Min(a.z, b.z)
		);
	}

	static DirectX::XMFLOAT3 Max(const DirectX::XMFLOAT3& a, const DirectX::XMFLOAT3& b)
	{
		return DirectX::XMFLOAT3(
			Max(a.x, b.x),
			Max(a.y, b.y),
			Max(a.z, b.z)
		);
	}

	static DirectX::XMFLOAT3 Ceil(const DirectX::XMFLOAT3& a, const DirectX::XMUINT3& b)
	{
		return DirectX::XMFLOAT3(ceil(a.x / b.x), ceil(a.y / b.y), ceil(a.z / b.z));
	}

	static DirectX::XMFLOAT3 Ceil(const DirectX::XMFLOAT3& a, const UINT& b)
	{
		return DirectX::XMFLOAT3(ceil(a.x / b), ceil(a.y / b), ceil(a.z / b));
	}

	static DirectX::XMUINT3 Ceil(const DirectX::XMUINT3& a, const DirectX::XMUINT3& b)
	{
		return DirectX::XMUINT3(ceil((float)a.x / b.x), ceil((float)a.y / b.y), ceil((float)a.z / b.z));
	}

	static DirectX::XMUINT3 Ceil(const DirectX::XMUINT3& a, const UINT& b)
	{
		return DirectX::XMUINT3(ceil(a.x / (float)b), ceil(a.y / (float)b), ceil(a.z / (float)b));
	}

	static DirectX::XMUINT2 Ceil(const DirectX::XMUINT2& a, const UINT& b)
	{
		return DirectX::XMUINT2(ceil(a.x / (float)b), ceil(a.y / (float)b));
	}

	static DirectX::XMFLOAT4 EulerToQuaternion(const DirectX::XMFLOAT3& eulerDir) {
		// Assuming you have a world up direction (usually {0.0f, 1.0f, 0.0f} for Y-axis up)
		DirectX::XMVECTOR forward = DirectX::XMVectorSet(eulerDir.x, eulerDir.y, eulerDir.z, 0.0f);
		DirectX::XMVECTOR up = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

		// Calculate the right and up vectors using cross product
		DirectX::XMVECTOR right = DirectX::XMVector3Cross(up, forward);
		up = DirectX::XMVector3Cross(forward, right);

		// Create the rotation matrix
		DirectX::XMMATRIX rotationMatrix = DirectX::XMMatrixLookToRH(DirectX::XMVectorZero(), forward, up);

		// Convert the matrix to a quaternion
		DirectX::XMVECTOR quaternion = DirectX::XMQuaternionRotationMatrix(rotationMatrix);

		// Store it in an XMFLOAT4 and return
		DirectX::XMFLOAT4 quaternionResult;
		DirectX::XMStoreFloat4(&quaternionResult, quaternion);
		return quaternionResult;
	}

	static DirectX::XMFLOAT3 QuaternionToEuler(DirectX::XMFLOAT4 quaternion)
	{
		// Extract quaternion components
		float x = quaternion.x;
		float y = quaternion.y;
		float z = quaternion.z;
		float w = quaternion.w;

		float roll, pitch, yaw;

		// Calculate the Euler angles
		// Pitch (x-axis rotation)
		float sinr_cosp = 2 * (w * x + y * z);
		float cosr_cosp = 1 - 2 * (x * x + y * y);
		pitch = std::atan2(sinr_cosp, cosr_cosp);

		// Yaw (y-axis rotation)
		float sinp = 2 * (w * y - z * x);
		if (std::abs(sinp) >= 1)
			yaw = std::copysign(DirectX::XM_PI / 2, sinp); // Use 90 degrees if out of range
		else
			yaw = std::asin(sinp);

		// Roll (z-axis rotation)
		float siny_cosp = 2 * (w * z + x * y);
		float cosy_cosp = 1 - 2 * (y * y + z * z);
		roll = std::atan2(siny_cosp, cosy_cosp);

		return DirectX::XMFLOAT3(pitch, yaw, roll);
	}

	template <typename T>
	static T Clamp(const T& x, const T& low, const T& high)
	{
		return x < low ? low : (x > high ? high : x);
	}

	template <typename T>
	static T Lerp(const T& a, const T& b, const float& t)
	{
		return a * (1.0f - t) + b * t;
	}

	template <typename T>
	static T MinMaxScale(const T& min, const T& max, const T& value)
	{
		return (value - min) / (max - min);
	}

	template <typename T>
	static float SmoothStep(const T& min, const T& max, const T& t)
	{
		T x = Clamp((t - min) / (max - min), 0.0f, 1.0f);
		return x * x * (3 - 2 * x);
	}

	template <typename T>
	static float Abs(const T& x)
	{
		return x >= 0 ? x : -x;
	}

	static float RandF()
	{
		return (float)(rand()) / (float)RAND_MAX;
	}

	static float RandF(float a, float b)
	{
		return a + RandF() * (b - a);
	}

	static int Rand(int a, int b)
	{
		return a + rand() % ((b - a) + 1);
	}

	// From MiniEngine
	template <typename T> 
	static __forceinline T DivideByMultiple(T value, size_t alignment)
	{
		return (T)((value + alignment - 1) / alignment);
	}


	static DirectX::XMVECTOR SphericalToCartesian(float radius, float theta, float phi)
	{
		return DirectX::XMVectorSet(
			radius * sinf(phi) * cosf(theta),
			radius * cosf(phi),
			radius * sinf(phi) * sinf(theta),
			1.0f
		);
	}

	static const float Infinity;

	static std::wstring MatrixToWstring(const DirectX::XMMATRIX& matrix);


	static std::wstring Matrix4x4ToWstring(const DirectX::XMFLOAT4X4& matrix);
	

};

