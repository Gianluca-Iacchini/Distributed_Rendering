#pragma once

#include "DirectXMath.h"
#include "vector"
#include <winsock2.h>
#include <ws2tcpip.h>

namespace CVGI
{
	class VoxelNetworkManager
	{
	public:
		VoxelNetworkManager();
		~VoxelNetworkManager();
		void Initialize();
		void SendVoxelInfo(DirectX::XMUINT3 voxelSize);
		void TransmitBuffer(std::vector<DirectX::XMUINT2> currentFrameRadianceData);

	private:
		SOCKET m_sockfd;
		struct sockaddr_in m_servAddr;
	};

}



