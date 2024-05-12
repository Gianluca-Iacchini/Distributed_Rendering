#include "DX12Lib/pch.h"
#include "Color.h"

using namespace DirectX;
using namespace DX12Lib;

Color Color::ToSRGB() const
{
    XMVECTOR T = XMVectorSaturate(m_color);
    XMVECTOR result = XMVectorSubtract(XMVectorScale(XMVectorPow(T, XMVectorReplicate(1.0f / 2.4f)), 1.055f), XMVectorReplicate(0.055f));
    result = XMVectorSelect(result, XMVectorScale(T, 12.92f), XMVectorLess(T, XMVectorReplicate(0.0031308f)));
    return XMVectorSelect(T, result, g_XMSelect1110);
}

Color Color::FromSRGB() const
{
    XMVECTOR T = XMVectorSaturate(m_color);
    XMVECTOR result = XMVectorPow(XMVectorScale(XMVectorAdd(T, XMVectorReplicate(0.055f)), 1.0f / 1.055f), XMVectorReplicate(2.4f));
    result = XMVectorSelect(result, XMVectorScale(T, 1.0f / 12.92f), XMVectorLess(T, XMVectorReplicate(0.0031308f)));
    return XMVectorSelect(T, result, g_XMSelect1110);
}
