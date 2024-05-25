#include "DX12Lib/pch.h"

#include "MathHelper.h"

const float MathHelper::Infinity = FLT_MAX;

std::wstring MathHelper::MatrixToWstring(const DirectX::XMMATRIX& matrix)
{
	std::wstringstream wss;
	DirectX::XMFLOAT4X4 mat;
	DirectX::XMStoreFloat4x4(&mat, matrix);
	wss << L"[" << mat._11 << L", " << mat._12 << L", " << mat._13 << L", " << mat._14 << L"]\n";
	wss << L"[" << mat._21 << L", " << mat._22 << L", " << mat._23 << L", " << mat._24 << L"]\n";
	wss << L"[" << mat._31 << L", " << mat._32 << L", " << mat._33 << L", " << mat._34 << L"]\n";
	wss << L"[" << mat._41 << L", " << mat._42 << L", " << mat._43 << L", " << mat._44 << L"]\n";
	return wss.str();
}

std::wstring MathHelper::Matrix4x4ToWstring(const DirectX::XMFLOAT4X4& matrix)
{
	std::wstringstream wss;
	wss << L"[" << matrix._11 << L", " << matrix._12 << L", " << matrix._13 << L", " << matrix._14 << L"]\n";
	wss << L"[" << matrix._21 << L", " << matrix._22 << L", " << matrix._23 << L", " << matrix._24 << L"]\n";
	wss << L"[" << matrix._31 << L", " << matrix._32 << L", " << matrix._33 << L", " << matrix._34 << L"]\n";
	wss << L"[" << matrix._41 << L", " << matrix._42 << L", " << matrix._43 << L", " << matrix._44 << L"]\n";
	return wss.str();
}
