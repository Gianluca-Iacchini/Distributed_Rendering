#pragma once

#include "vector"
#include "memory"
#include "string"
#include "thread"
#include "functional"
#include "queue"
#include "mutex"


#include "../extern/enet/include/enet/enet.h"


namespace Commons
{
	template<typename T>
	class ReusableQueue {
	public:
		ReusableQueue() {}

		void Push(T element) {
			std::lock_guard<std::mutex> lock(m_mutex);
			m_elementQueue.push(element);
			m_cv.notify_one();
		}

		void GetFromPool(T& element) {
			std::unique_lock<std::mutex> lock(m_mutex);

			// If the flag is set, we wait for an element to be available in the pool
			if (ShouldWait)
			{
				m_cv.wait(lock, [this] { return !m_availableElementsPool.empty() || m_done; });
			}
			// If the flag is not set and the pool is empty, we return the oldest element in the queue (the one at the front) to the pool 
			else if (m_availableElementsPool.empty())
			{
				m_availableElementsPool.push_back(m_elementQueue.front());
				m_elementQueue.pop();
			}

			// Usually needed only when the thread has to be abruptly stopped
			if (m_done)
				return;

			element = m_availableElementsPool.back();
			m_availableElementsPool.pop_back();
		}

		void Pop(T& element) {
			std::unique_lock<std::mutex> lock(m_mutex);
			m_cv.wait(lock, [this] { return !m_elementQueue.empty() || m_done; });

			// Usually needed only when the thread has to be abruptly stopped
			if (m_done)
				return;

			element = m_elementQueue.front();
			m_elementQueue.pop();
		}

		bool NonBlockingPop(T& element) {
			std::unique_lock<std::mutex> lock(m_mutex);

			if (m_elementQueue.empty())
				return false;

			element = m_elementQueue.front();
			m_elementQueue.pop();
			return true;
		}

		void AddNewElementToPool(T element) {
			std::lock_guard<std::mutex> lock(m_mutex);
			m_availableElementsPool.push_back(element);
		}

		void ReturnToPool(T& element) {
			std::lock_guard<std::mutex> lock(m_mutex);

			m_availableElementsPool.push_back(element);
		}

		void SetDone(bool done) {
			std::unique_lock<std::mutex> lock(m_mutex);
			m_done = done;
			if (m_done)
				m_cv.notify_all();
		}

		unsigned int GetQueueSize() {
			std::unique_lock<std::mutex> lock(m_mutex);
			return m_elementQueue.size();
		}

		unsigned int GetPoolSize() {
			std::unique_lock<std::mutex> lock(m_mutex);
			return m_availableElementsPool.size();
		}

		std::mutex& GetMutex() {
			return m_mutex;
		}

		std::queue<T>& GetQueue() {
			return m_elementQueue;
		}

		std::vector<T>& GetPool() { return m_availableElementsPool; }

	public:
		bool ShouldWait = true;

	private:
		int m_maxElements = 0;
		std::queue<T> m_elementQueue;
		std::vector<T> m_availableElementsPool;
		std::mutex m_mutex;
		std::condition_variable m_cv;
		bool m_done = false;
	};
}

enum class NetworkHostType
{
	Server,
	Client,
	None,
};

namespace Commons
{
	class NetworkHost;

	struct NetworkPacket
	{
		friend class NetworkHost;

	private:
		NetworkPacket() = default;

	public:
		enum class PacketType
		{
			PACKET_RELIABLE = 0,
			PACKET_UNRELIABLE = 1,
		};

		// Utility to append data of any type to a buffer
		template <typename T>
		void AppendToBuffer(const T& value) {
			const std::uint8_t* dataPtr = reinterpret_cast<const std::uint8_t*>(&value);
			m_data.insert(m_data.end(), dataPtr, dataPtr + sizeof(T));
		}

		void AppendToBuffer(const std::uint8_t* data, std::size_t size) {
			m_data.insert(m_data.end(), data, data + size);
		}

		void AppendToBuffer(const std::string& str) {
			m_data.insert(m_data.end(), str.begin(), str.end());
		}

		void AppendToBuffer(const char* str) {
			m_data.insert(m_data.end(), str, str + strlen(str) + 1);
		}

		void AppendToBuffer(const std::vector<std::uint8_t>& data) {
			m_data.insert(m_data.end(), data.begin(), data.end());
		}

		void AppendToBuffer(const std::vector<UINT32>& data) {
			std::size_t currentSize = m_data.size();
			std::size_t appendDataByteSize = data.size() * sizeof(UINT32);

			m_data.resize(currentSize + appendDataByteSize);
			memcpy(m_data.data() + currentSize, data.data(), appendDataByteSize);
		}

		void SetPacketType(PacketType type) { m_packetType = type; }
		PacketType GetPacketType() const { return m_packetType; }

		void ClearPacket() { m_data.clear(); }
		void SetData(const std::vector<std::uint8_t>& data) { m_data = data; }
		const uint8_t* GetData() const { return m_data.data(); }
		std::size_t GetSize() const { return m_data.size(); }
		const std::vector<std::uint8_t>& GetDataVector() const { return m_data; }

	private:
		// shared_ptr needs an accessible constructor to instantiate the object.
		// We use this little trick to keep the constructor private and still allow shared_ptr to instantiate the class
		// https://stackoverflow.com/questions/8147027/how-do-i-call-stdmake-shared-on-a-class-with-only-protected-or-private-const/8147213#8147213
		static std::shared_ptr<NetworkPacket> MakeShared()
		{
			struct make_shared_enabler : public NetworkPacket {};

			return std::make_shared<make_shared_enabler>();
		}

	private:
		std::vector<std::uint8_t> m_data;
		PacketType m_packetType = PacketType::PACKET_RELIABLE;
	};

	struct PacketGuard
	{
		friend class NetworkHost;

	private:

		std::function<void(std::shared_ptr<NetworkPacket>)> m_deleter;

		PacketGuard(std::shared_ptr<NetworkPacket> packet, std::function<void(std::shared_ptr<NetworkPacket>)> deleter)
			: m_packet(std::move(packet)), m_deleter(deleter) {}

	public:

		~PacketGuard() { if (m_packet == nullptr || !m_isMovedToPool) { m_deleter(m_packet); } }
		NetworkPacket* operator->() { return m_packet.get(); }
		NetworkPacket& operator*() { return *m_packet; }

		operator NetworkPacket* () { return m_packet.get(); }

		operator std::shared_ptr<NetworkPacket>() { return m_packet; }

	private:
		std::shared_ptr<NetworkPacket> m_packet;
		bool m_isMovedToPool = false;
	};

	class NetworkHost
	{
	public:
		NetworkHost();
		virtual ~NetworkHost();


		virtual void Connect(const char* address, const std::uint16_t port);
		virtual void StartServer(const std::uint16_t port);
		virtual void Disconnect();

		
		void SendData(PacketGuard& packet);
		PacketGuard CreatePacket();

		bool IsConnected() const { return m_isConnected; }
		bool HasPeers() const;
		std::string GetPeerAddress() const;
		UINT32 GetPing();

		static bool CheckPacketHeader(const NetworkPacket* packet, const std::string& prefix);

		std::string GetHostAddress() const;

		void SetDefaultCompressionLevel(std::uint8_t level) { m_defaultCompressionLevel = level; }
		std::uint8_t GetDefaultCompressionLevel() const { return m_defaultCompressionLevel; }

		float GetAverageCompressionRatio() const;
		float GetAverageCompressionTime() const;

		void ResetCompressionStats() { m_nCompressions = 0; m_totalCompressionRatio = 0.0f; m_totalCompressionTime = 0.0f; }

	protected:
		virtual void MainNetworkLoop();
		virtual void SendDataLoop();
		virtual void ReceiveDataLoop();

		virtual void InitializeAsServer(uint16_t port);
		virtual void InitializeAsClient();

	private:
		void PrintNetworkInterfaces();
		virtual bool CompressData(NetworkPacket* packet);
		virtual bool DecompressData(NetworkPacket* packet);

	public:
		static void InitializeEnet();
		static void DeinitializeEnet();
		static UINT64 GetEpochTime();

	public:
		std::function<void(const ENetPeer*)> OnPeerConnected;
		std::function<void(const NetworkPacket*)> OnPacketReceived;
		std::function<void(const ENetPeer*)> OnPeerDisconnected;

	private:

		ENetHost* m_host = nullptr;
		ENetAddress m_address;
		ENetPeer* m_Peer = nullptr;

		NetworkHostType m_hostType;
		bool m_isConnected = false;

		std::thread m_mainNetworkThread;

		std::string m_hostAddress;

		// Using raw pointer is probably more efficient, but shared_ptr is safer and easier to manage
		ReusableQueue<std::shared_ptr<NetworkPacket>> m_packetsToSend;
		ReusableQueue<std::shared_ptr<NetworkPacket>> m_receivedPackets;

		int m_defaultCompressionLevel = 3;

		UINT64 m_nCompressions = 0;
		float m_totalCompressionRatio = 0.0f;
		float m_totalCompressionTime = 0.0f;


	private:
		static bool m_isEnetInitialized;
	};
}



