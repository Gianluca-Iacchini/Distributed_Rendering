#include "NetworkManager.h"
#include "minmax.h"

#include "WS2tcpip.h"
#include "in6addr.h"
#include "chrono"
#include "zstd.h"

#ifdef CVGI_GL
#include "../StreamingClient/Helpers.h"
#endif // CVGI_GL

#define QUEUE_SIZE 3

void Commons::NetworkHost::InitializeEnet()
{
	if (!m_isEnetInitialized)
	{
		if (enet_initialize() != 0)
		{
			DXLIB_CORE_ERROR("An error occurred while initializing ENet");
			return;
		}

		m_isEnetInitialized = true;
	}
}

void Commons::NetworkHost::DeinitializeEnet()
{
	if (m_isEnetInitialized)
	{
		enet_deinitialize();
		m_isEnetInitialized = false;
	}

}

void Commons::NetworkHost::InitializeAsClient()
{

	if (!m_isEnetInitialized)
	{
		DXLIB_CORE_ERROR("ENet is not initialized");
		return;
	}

	if (m_host != nullptr)
	{
		enet_host_destroy(m_host);
	}

	m_host = enet_host_create(nullptr, 1, 1, 0, 0);

	char addrStr[INET6_ADDRSTRLEN];
	if (enet_address_get_host_ip(&m_host->address, addrStr, sizeof(addrStr)) != 0)
	{
		DXLIB_CORE_ERROR(L"Failed to retrieve host address");
		return;
	}

	m_hostAddress = std::string(addrStr);

	if (m_host == NULL)
	{
		DXLIB_CORE_ERROR(L"Failed to initialize host as client");
		return;
	}

	DXLIB_CORE_INFO("Host successfully initialized as client");

	m_hostType = NetworkHostType::Client;
}

void Commons::NetworkHost::Disconnect()
{
	if (m_mainNetworkThread.joinable())
	{

		m_isConnected = false;

		m_mainNetworkThread.join();

		if (m_Peer != nullptr)
		{
			bool successfullyDisconnected = false;
			ENetEvent event;
			enet_peer_disconnect(m_Peer, 0);

			while (enet_host_service(m_host, &event, 1500) > 0)
			{
				switch (event.type)
				{
				case ENET_EVENT_TYPE_RECEIVE:
					enet_packet_destroy(event.packet);
					break;

				case ENET_EVENT_TYPE_DISCONNECT:
					successfullyDisconnected = true;
					break;
				}
			}

			if (successfullyDisconnected)
			{
				DXLIB_CORE_INFO("Successfully disconnected from peer");
			}
			else
			{
				DXLIB_CORE_ERROR("Failed to disconnect from peer");
				enet_peer_reset(m_Peer);
			}
		}
	}

	m_Peer = nullptr;
}


void Commons::NetworkHost::InitializeAsServer(uint16_t port)
{
	if (!m_isEnetInitialized)
	{
		DXLIB_CORE_ERROR("ENet is not initialized");
		return;
	}

	if (m_hostType != NetworkHostType::None)
	{
		DXLIB_CORE_ERROR("Host is already initialized");
		return;
	}

	m_address.host = ENET_HOST_ANY;
	m_address.port = port;

	m_host = enet_host_create(&m_address, 1, 1, 0, 0);

	char addrStr[INET6_ADDRSTRLEN];
	if (enet_address_get_host_ip(&m_host->address, addrStr, sizeof(addrStr)) != 0)
	{
		DXLIB_CORE_ERROR(L"Failed to retrieve host address");
		return;
	}

	m_hostAddress = std::string(addrStr);

	if (m_host == NULL)
	{
		DXLIB_CORE_ERROR(L"Failed to initialize host as server");
		return;
	}

	DXLIB_CORE_INFO("Host successfully initialized as server");

	m_hostType = NetworkHostType::Server;
}

Commons::NetworkHost::NetworkHost() : m_hostType(NetworkHostType::None)
{
	m_receivedPackets.ShouldWait = false;
}

Commons::NetworkHost::~NetworkHost()
{
	Disconnect();

	if (m_host != nullptr)
	{
		enet_host_destroy(m_host);
		m_host = nullptr;
	}
}

void Commons::NetworkHost::Connect(const char* address, const std::uint16_t port)
{

	if (m_hostType == NetworkHostType::Server)
	{
		DXLIB_CORE_ERROR("Only clients can initiate a connection");
		return;
	}

	this->InitializeAsClient();

	if (enet_address_set_host(&m_address, address) < 0)
	{
		DXLIB_CORE_ERROR("Error at setting host");
		return;
	}
	m_address.port = port;

	ENetPeer* peer = enet_host_connect(m_host, &m_address, 1, 0);

	if (peer == NULL)
	{
		DXLIB_CORE_ERROR("Failed to connect to peer {0} at port {1}", address, port);
		return;
	}


	m_mainNetworkThread = std::thread(&NetworkHost::MainNetworkLoop, this);
}

void Commons::NetworkHost::StartServer(const std::uint16_t port)
{
	if (m_hostType != NetworkHostType::Server)
	{
		if (m_hostType == NetworkHostType::None)
		{
			this->InitializeAsServer(port);
		}
		else
		{
			DXLIB_CORE_ERROR("Only clients can initiate a connection");
			return;
		}
	}

	PrintNetworkInterfaces();

	m_mainNetworkThread = std::thread(&NetworkHost::MainNetworkLoop, this);
}

void Commons::NetworkHost::SendData(Commons::PacketGuard& packet)
{
	packet.m_isMovedToPool = true;
	m_packetsToSend.Push(packet.m_packet);
}


Commons::PacketGuard Commons::NetworkHost::CreatePacket()
{
	// If the pool is empty, add a new element, an old packet could be returned to the pool between this call and the GetFromPool call
	// Since we are not holding the mutex between these two calls, but that's okay.
	if (m_packetsToSend.GetPoolSize() == 0)
	{
		m_packetsToSend.AddNewElementToPool(NetworkPacket::MakeShared());
	}

	std::shared_ptr<NetworkPacket> newPacket = nullptr;
	m_packetsToSend.GetFromPool(newPacket);

	bool packetToDiscard = false;
	// This should only happen if the client is disconnected while the packet is being created
	if (newPacket == nullptr)
	{
		newPacket = NetworkPacket::MakeShared();
		packetToDiscard = true;
	}

	PacketGuard packetGuard = PacketGuard(newPacket, [this, packetToDiscard](std::shared_ptr<NetworkPacket> packet)
		{
			if (!packetToDiscard)
			{
				DXLIB_CORE_WARN("Packet was not disposed of properly.");
				m_packetsToSend.ReturnToPool(packet);
			}
		});

	packetGuard->SetPacketType(NetworkPacket::PacketType::PACKET_RELIABLE);

	return packetGuard;
}

bool Commons::NetworkHost::HasPeers() const
{
	return m_Peer != nullptr;
}

std::string Commons::NetworkHost::GetPeerAddress() const
{
	if (m_Peer == nullptr)
	{
		return std::string("");
	}

	char addrStr[INET6_ADDRSTRLEN];
	if (enet_address_get_host_ip(&m_Peer->address, addrStr, sizeof(addrStr)) != 0)
	{
		DXLIB_CORE_ERROR(L"Failed to retrieve host address");
		return std::string("");
	}

	return std::string(addrStr);
}

UINT32 Commons::NetworkHost::GetPing()
{
	if (m_Peer != nullptr)
	{
		return m_Peer->roundTripTime;
	}

	return 0;
}

bool Commons::NetworkHost::CheckPacketHeader(const NetworkPacket* packet, const std::string& prefix)
{
	auto& data = packet->GetDataVector();

	if (data.size() < prefix.size()) {
		return false;
	}

	// Temporary buffer to hold the prefix-sized substring from data
	std::vector<char> temp(prefix.size(), 0);
	memcpy(temp.data(), data.data(), prefix.size());

	return strncmp(temp.data(), prefix.c_str(), prefix.size()) == 0;

}

std::string Commons::NetworkHost::GetHostAddress() const
{
	return m_hostAddress;
}

float Commons::NetworkHost::GetAverageCompressionRatio() const
{
	return m_totalCompressionRatio / max(m_nCompressions, 1);
}

float Commons::NetworkHost::GetAverageCompressionTime() const
{
	return m_totalCompressionTime / max(m_nCompressions, 1);
}

void Commons::NetworkHost::PrintNetworkInterfaces()
{

}

bool Commons::NetworkHost::CompressData(NetworkPacket* packet)
{
	auto& packetData = packet->GetDataVector();
	size_t prevSize = packetData.size();

	size_t compressBound = ZSTD_compressBound(packetData.size());

	std::vector<uint8_t> compressedData(compressBound);


	int compressionLevel = m_defaultCompressionLevel;

	if (prevSize < 200)
	{
		compressionLevel = 0;
	}

	// Get time now
	auto start = std::chrono::high_resolution_clock::now();

	size_t compressedSize = ZSTD_compress(
		compressedData.data(),
		compressedData.size(),
		packetData.data(),
		packetData.size(),
		compressionLevel
	);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	m_totalCompressionTime += duration.count() / 1000.0f;

	m_totalCompressionRatio += (float)prevSize / (float)compressedSize;
	m_nCompressions += 1;

	if (ZSTD_isError(compressedSize)) {
		DXLIB_CORE_ERROR("Zstd compression failed: " + std::string(ZSTD_getErrorName(compressedSize)));
		return false;
	}

	compressedData.resize(compressedSize);

	packet->ClearPacket();
	packet->SetData(compressedData);

	return true;
}

bool Commons::NetworkHost::DecompressData(NetworkPacket* packet)
{
	auto& packetData = packet->GetDataVector();

	size_t prevSize = packetData.size();

	size_t decompressedSize = ZSTD_getFrameContentSize(packetData.data(), packetData.size());

	if (decompressedSize == ZSTD_CONTENTSIZE_ERROR) {
		DXLIB_CORE_ERROR("Zstd decompression failed: " + std::string(ZSTD_getErrorName(decompressedSize)));
		return false;
	}

	std::vector<uint8_t> decompressedData(decompressedSize);

	decompressedSize = ZSTD_decompress(
		decompressedData.data(),
		decompressedData.size(),
		packetData.data(),
		packetData.size()
	);

	if (ZSTD_isError(decompressedSize)) {
		DXLIB_CORE_ERROR("Zstd decompression failed: " + std::string(ZSTD_getErrorName(decompressedSize)));
		return false;
	}

	decompressedData.resize(decompressedSize);

	packet->ClearPacket();
	packet->SetData(decompressedData);

	return true;
}

void Commons::NetworkHost::MainNetworkLoop()
{
	DXLIB_CORE_INFO("Listening for messages from incoming connections.");

	ENetEvent receivedEvent;

	m_receivedPackets.SetDone(false);
	m_packetsToSend.SetDone(false);

	for (UINT i = 0; i < QUEUE_SIZE; i++)
	{
		m_packetsToSend.AddNewElementToPool(NetworkPacket::MakeShared());
		m_receivedPackets.AddNewElementToPool(NetworkPacket::MakeShared());
	}

	m_isConnected = true;

	std::thread sendDataThread(&NetworkHost::SendDataLoop, this);
	std::thread receiveDataThread(&NetworkHost::ReceiveDataLoop, this);

	struct in6_addr addr;
	char addrStr[INET6_ADDRSTRLEN];

	while (m_isConnected)
	{
		int serviceResult = enet_host_service(m_host, &receivedEvent, 0);
		if (serviceResult > 0)
		{
			if (inet_ntop(AF_INET6, &receivedEvent.peer->address.host, addrStr, INET6_ADDRSTRLEN) == NULL)
			{
				DXLIB_CORE_ERROR("Failed to convert host address to string");
				continue;
			}

			if (receivedEvent.type == ENET_EVENT_TYPE_CONNECT)
			{
				DXLIB_CORE_INFO("A new client connected from {0}:{1}", addrStr, receivedEvent.peer->address.port);
				m_Peer = receivedEvent.peer;
				if (m_hostType == NetworkHostType::Client)
				{
					m_isConnected = true;
				}
				if (OnPeerConnected)
				{
					OnPeerConnected(receivedEvent.peer);
				}

			}
			else if (receivedEvent.type == ENET_EVENT_TYPE_RECEIVE)
			{
				std::shared_ptr<NetworkPacket> receivedPacket;
				m_receivedPackets.GetFromPool(receivedPacket);
				receivedPacket->ClearPacket();
				receivedPacket->AppendToBuffer(receivedEvent.packet->data, receivedEvent.packet->dataLength);
				m_receivedPackets.Push(receivedPacket);
				enet_packet_destroy(receivedEvent.packet);
			}
			else if (receivedEvent.type == ENET_EVENT_TYPE_DISCONNECT)
			{
				DXLIB_CORE_INFO("Client disconnected from {0}:{1}", addrStr, receivedEvent.peer->address.port);
				if (OnPeerDisconnected)
					OnPeerDisconnected(receivedEvent.peer);

				enet_peer_disconnect(receivedEvent.peer, 0);
				if (m_hostType == NetworkHostType::Client)
				{
					m_isConnected = false;
				}
				m_Peer = nullptr;

			}
		}
		else if (serviceResult < 0)
		{

			DXLIB_CORE_ERROR("Error while receiving data");
		}
		// Else no event occurred
	}

	m_receivedPackets.SetDone(true);
	m_packetsToSend.SetDone(true);

	if (sendDataThread.joinable())
		sendDataThread.join();

	if (receiveDataThread.joinable())
		receiveDataThread.join();
}

void Commons::NetworkHost::SendDataLoop()
{
	std::shared_ptr<NetworkPacket> packet = nullptr;

	while (m_isConnected)
	{
		// This will block until a packet is available
		m_packetsToSend.Pop(packet);

		if (m_Peer == nullptr)
		{
			DXLIB_CORE_WARN("A peer was not set. Packets will be discarded");
			continue;
		}


		if (packet == nullptr || packet->m_data.size() == 0)
		{
			DXLIB_CORE_WARN("Packet contains no data");
			continue;
		}


		if (CompressData(packet.get()))
		{
			ENetPacketFlag flags = ENET_PACKET_FLAG_RELIABLE;

			if (packet->GetPacketType() == NetworkPacket::PacketType::PACKET_UNRELIABLE)
			{
				flags = ENET_PACKET_FLAG_UNRELIABLE_FRAGMENT;
			}

			ENetPacket* enetPacket = enet_packet_create(packet->m_data.data(), packet->m_data.size(), flags);

			if (enet_peer_send(m_Peer, 0, enetPacket) < 0)
			{
				DXLIB_CORE_ERROR("Error while sending packet");
			}
		}

		m_packetsToSend.ReturnToPool(packet);
	}
}

void Commons::NetworkHost::ReceiveDataLoop()
{
	std::shared_ptr<NetworkPacket> packet = nullptr;
	while (m_isConnected)
	{
		// This will block until a packet is available
		m_receivedPackets.Pop(packet);

		if (packet == nullptr)
		{
			DXLIB_CORE_WARN("Received packet was null");
			continue;
		}

		if (OnPacketReceived)
		{
			NetworkPacket* netPacket = packet.get();
			if (DecompressData(netPacket))
			{
				OnPacketReceived(netPacket);
			}
		}


		m_receivedPackets.ReturnToPool(packet);
	}
}

bool Commons::NetworkHost::m_isEnetInitialized = false;

Commons::PacketGuard::PacketGuard(std::shared_ptr<NetworkPacket> packet, std::function<void(std::shared_ptr<NetworkPacket>)> deleter)
	: m_packet(std::move(packet)), m_deleter(deleter) {}

Commons::PacketGuard::PacketGuard(PacketGuard& other)
{
	m_packet = other.m_packet;
	m_deleter = other.m_deleter;
	m_isMovedToPool = other.m_isMovedToPool;
	other.m_isMovedToPool = true;
}

Commons::PacketGuard::~PacketGuard()
{
	if (m_packet == nullptr || !m_isMovedToPool) 
	{
		m_deleter(m_packet);
	}
}
