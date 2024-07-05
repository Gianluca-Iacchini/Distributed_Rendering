#pragma once

#include <DirectXMath.h>
#include <string>

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

