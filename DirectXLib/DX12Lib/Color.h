#ifndef COLOR_H
#define COLOR_H
#include "Helpers.h"

class Color
{
public:
	Color() : m_color(DirectX::g_XMOne) {}
	Color(DirectX::FXMVECTOR vec) { m_color.v = vec; };
	Color(const DirectX::XMVECTORF32& vec) { m_color = vec; };
	Color(float r, float g, float b, float a = 1.0f) { m_color.v = DirectX::XMVectorSet(r, g, b, a); };
	Color(std::uint16_t r, std::uint16_t g, std::uint16_t b, std::uint16_t a = 255, std::uint16_t depth = 8)
	{
		m_color.v = DirectX::XMVectorScale(DirectX::XMVectorSet(r, g, b, a), 1.0f / ((1 << depth) - 1));
	}

	explicit Color(std::uint32_t rgba)
	{
		float r = (float)((rgba >> 0) & 0xFF);
		float g = (float)((rgba >> 8) & 0xFF);
		float b = (float)((rgba >> 16) & 0xFF);
		float a = (float)((rgba >> 24) & 0xFF);

		m_color.v = DirectX::XMVectorScale(DirectX::XMVectorSet(r, g, b, a), 1.0f / 255.0f);
	}

	float R() const { return DirectX::XMVectorGetX(m_color); }
	float G() const { return DirectX::XMVectorGetY(m_color); }
	float B() const { return DirectX::XMVectorGetZ(m_color); }
	float A() const { return DirectX::XMVectorGetW(m_color); }

	bool operator==(const Color& rhs) const { return DirectX::XMVector4Equal(m_color, rhs.m_color); }
	bool operator!=(const Color& rhs) const { return !DirectX::XMVector4Equal(m_color, rhs.m_color); }

	void SetR(float r) { m_color.f[0] = r; }
	void SetG(float g) { m_color.f[1] = g; }
	void SetB(float b) { m_color.f[2] = b; }
	void SetA(float a) { m_color.f[3] = a; }
	void SetRGB(float r, float g, float b) { m_color.v = DirectX::XMVectorSelect(m_color, DirectX::XMVectorSet(r, g, b, b), DirectX::g_XMMask3); }

	Color ToSRGB() const;
	Color FromSRGB() const;

public:
	inline static Color Max(Color a, Color b)
	{
		return Color(DirectX::XMVectorMax(a.m_color, b.m_color));
	}

	inline static Color Min(Color a, Color b)
	{
		return Color(DirectX::XMVectorMin(a.m_color, b.m_color));
	}

	inline static Color Clamp(Color x, Color a, Color b)
	{
		return Color(DirectX::XMVectorClamp(x.m_color, a.m_color, b.m_color));
	}

private:
	DirectX::XMVECTORF32 m_color;

public:
	float* GetPtr() { return reinterpret_cast<float*>(this); }
	float& operator[](size_t index) { return GetPtr()[index]; }
};

#endif // !COLOR_H



