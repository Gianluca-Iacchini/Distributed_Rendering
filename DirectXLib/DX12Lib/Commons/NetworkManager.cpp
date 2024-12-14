#include "DX12Lib/pch.h"
#include "NetworkManager.h"

#define ENET_DEBUG 1
#include "WS2tcpip.h"
#include "in6addr.h"
#include "../extern/enet/include/enet/enet.h"

#define QUEUE_SIZE 3

using namespace DX12Lib;

void DX12Lib::NetworkHost::InitializeEnet()
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

void DX12Lib::NetworkHost::DeinitializeEnet()
{
	if (m_isEnetInitialized)
	{
		enet_deinitialize();
		m_isEnetInitialized = false;
	}

}

void DX12Lib::NetworkHost::InitializeAsClient()
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

	m_host = enet_host_create(nullptr, 1, 1, 0, 0);
	

	if (m_host == NULL)
	{
		DXLIB_CORE_ERROR(L"Failed to initialize host as client");
		return;
	}

	DXLIB_CORE_INFO("Host successfully initialized as client");

	m_hostType = NetworkHostType::Client;
}

void DX12Lib::NetworkHost::Disconnect()
{
	if (m_mainNetworkThread.joinable())
	{
		if (m_isConnected)
		{
			m_isConnected = false;
		}

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

	

		if (m_host != nullptr)
		{
			enet_host_destroy(m_host);
			m_host = nullptr;
		}

		m_Peer = nullptr;
	}
}


void DX12Lib::NetworkHost::InitializeAsServer(uint16_t port)
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

	if (m_host == NULL)
	{
		DXLIB_CORE_ERROR(L"Failed to initialize host as server");
		return;
	}

	DXLIB_CORE_INFO("Host successfully initialized as server");

	m_hostType = NetworkHostType::Server;
}

DX12Lib::NetworkHost::NetworkHost() : m_hostType(NetworkHostType::None)
{
	m_receivedPackets.ShouldWait = false;

	for (UINT i = 0; i < QUEUE_SIZE; i++)
	{
		m_packetsToSend.AddNewElementToPool(NetworkPacket::MakeShared());
		m_receivedPackets.AddNewElementToPool(NetworkPacket::MakeShared());
	}
}

DX12Lib::NetworkHost::~NetworkHost()
{
	Disconnect();
}

void DX12Lib::NetworkHost::Connect(const std::string address, const std::uint16_t port)
{
	
	if (m_hostType != NetworkHostType::Client)
	{
		if (m_hostType == NetworkHostType::None)
		{
			this->InitializeAsClient();
		}
		else
		{
			DXLIB_CORE_ERROR("Only clients can initiate a connection");
			return;
		}
	}

	enet_address_set_host(&m_address, address.c_str());
	m_address.port = port;

	m_Peer = enet_host_connect(m_host, &m_address, 1, 0);

	if (m_Peer == NULL)
	{
		DXLIB_CORE_ERROR("Failed to connect to peer {0} at port {1}", address, port);
		return;
	}


	m_mainNetworkThread = std::thread(&NetworkHost::MainNetworkLoop, this);
}

void DX12Lib::NetworkHost::StartServer(const std::uint16_t port)
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

	m_mainNetworkThread = std::thread(&NetworkHost::MainNetworkLoop, this);
}

void DX12Lib::NetworkHost::SendData(DX12Lib::PacketGuard& packet)
{
	packet.m_isMovedToPool = true;
	m_packetsToSend.Push(packet.m_packet);
}

//void DX12Lib::NetworkHost::OnPacketReceived(const NetworkPacket* packet)
//{
//	auto& dataVector = packet->GetDataVector();
//
//	std::uint8_t message[5] = { 0 };
//	memcpy(&message, dataVector.data(), 5);
//
//	DXLIB_CORE_INFO("Received message: [{0},{1},{2},{3},{4}] of length {5}", message[0], message[1], message[2], message[3], message[4], dataVector.size());
//}

PacketGuard DX12Lib::NetworkHost::CreatePacket()
{
	// If the pool is empty, add a new element, an old packet could be returned to the pool between this call and the GetFromPool call
	// Since we are not holding the mutex between these two calls, but that's okay.
	if (m_packetsToSend.GetPoolSize() == 0)
	{
		m_packetsToSend.AddNewElementToPool(NetworkPacket::MakeShared());
	}

	std::shared_ptr<NetworkPacket> newPacket = nullptr;
	m_packetsToSend.GetFromPool(newPacket);

	assert(newPacket != nullptr && "Created Packed was null");

	PacketGuard packetGuard = PacketGuard(newPacket, [this](std::shared_ptr<NetworkPacket> packet)
		{
			DXLIB_CORE_INFO("Returning packet to pool");
			m_packetsToSend.ReturnToPool(packet);
		});

	return packetGuard;
}

bool DX12Lib::NetworkHost::CheckPacketHeader(const NetworkPacket* packet, const std::string& prefix)
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

void DX12Lib::NetworkHost::MainNetworkLoop()
{
	DXLIB_CORE_INFO("Listening for messages from incoming connections.");

	ENetEvent receivedEvent;

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
				if (OnPeerConnected)
					OnPeerConnected(receivedEvent.peer);
			}
			else if (receivedEvent.type == ENET_EVENT_TYPE_RECEIVE)
			{
				DXLIB_CORE_INFO("Packet received from {0}:{1}", addrStr, receivedEvent.peer->address.port);
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
				m_isConnected = false;
				m_Peer = nullptr;
			}
		}
		else if (serviceResult < 0)
		{
			
			DXLIB_CORE_ERROR("Error while receiving data");
		}
		// Else no event occurred
	}

	m_receivedPackets.SetDone();
	m_packetsToSend.SetDone();

	if (sendDataThread.joinable())
		sendDataThread.join();
	
	if (receiveDataThread.joinable())
		receiveDataThread.join();
}

void DX12Lib::NetworkHost::SendDataLoop()
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

		ENetPacket* enetPacket = enet_packet_create(packet->m_data.data(), packet->m_data.size(), ENET_PACKET_FLAG_RELIABLE);

		if (enet_peer_send(m_Peer, 0, enetPacket) < 0)
		{
			DXLIB_CORE_ERROR("Error while sending packet");
		}
				
		m_packetsToSend.ReturnToPool(packet);
	}
}

void DX12Lib::NetworkHost::ReceiveDataLoop()
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
			OnPacketReceived(packet.get());

		m_receivedPackets.ReturnToPool(packet);
	}
}

bool DX12Lib::NetworkHost::m_isEnetInitialized = false;

