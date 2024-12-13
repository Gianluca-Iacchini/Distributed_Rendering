#include "VoxelNetworkManager.h"
#include "DX12Lib/pch.h"

CVGI::VoxelNetworkManager::VoxelNetworkManager()
{
	m_sockfd = 0;
	m_servAddr = {};
}

CVGI::VoxelNetworkManager::~VoxelNetworkManager()
{
	if (m_sockfd != INVALID_SOCKET) {
		closesocket(m_sockfd);
		m_sockfd = INVALID_SOCKET;
	}

	WSACleanup();
}

void CVGI::VoxelNetworkManager::Initialize()
{
	WSADATA wsaData;

	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		DXLIB_CORE_ERROR("WSAStartup failed.");
		return;
	}

	m_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	if (m_sockfd == INVALID_SOCKET) {
		DXLIB_CORE_ERROR("Socket creation failed.");
		WSACleanup();
		return;
	}

	memset(&m_servAddr, 0, sizeof(m_servAddr));

	m_servAddr.sin_family = AF_INET;
	m_servAddr.sin_port = htons(12345);
	inet_pton(AF_INET, "127.0.0.1", &m_servAddr.sin_addr);

	DXLIB_CORE_ERROR("Winsock initialized");
}

void CVGI::VoxelNetworkManager::SendVoxelInfo(DirectX::XMUINT3 voxelSize)
{
	std::vector<uint8_t> voxelInfoMessage(3);
	voxelInfoMessage[0] = voxelSize.x;
	voxelInfoMessage[1] = voxelSize.y;
	voxelInfoMessage[2] = voxelSize.z;

	if (sendto(m_sockfd, reinterpret_cast<const char*>(voxelInfoMessage.data()), voxelInfoMessage.size(), 0, (struct sockaddr*)&m_servAddr, sizeof(m_servAddr)) == SOCKET_ERROR)
	{
		int error = WSAGetLastError();
		DXLIB_CORE_ERROR("Failed to send message {0}", error);
		return;
	}
}

void CVGI::VoxelNetworkManager::TransmitBuffer(std::vector<DirectX::XMUINT2> currentFrameRadianceData)
{
	if (sendto(m_sockfd, reinterpret_cast<const char*>(currentFrameRadianceData.data()), currentFrameRadianceData.size() * 2, 0, (struct sockaddr*)&m_servAddr, sizeof(m_servAddr)) == SOCKET_ERROR)
	{
		int error = WSAGetLastError();
		DXLIB_CORE_ERROR("Failed to send message {0}", error);
		return;
	}
}
