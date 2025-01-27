#include <iostream>
#include "StreamRenderer.h"

#include "Helpers.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "FFmpegDemuxer.h"
#include "NVDecoder.h"
#include "ColorSpace.h"

#include "NetworkManager.h"

#include "imgui.h"

#include <numeric>

#include "UIHelpers.h"


float lastTime = 0.0f;
float currentTime = 0.0f;
float accumulatedTime = 0.0f;

bool g_isDecodeDone = false;


double lastX = 0.0;
double lastY = 0.0;
std::string lastMouseXTruncated = "0";
std::string lastMouseYTruncated = "0";
bool firstMouse = true;

int cameraForwardValue = 0;
int cameraStrafeValue = 0;
int cameraLiftValue = 0;

Commons::NetworkHost m_clientHost;


std::uint8_t m_movementInputBitmask;
std::uint8_t m_lightInputBitmask;

float m_mousePosX = 0.0f;
float m_mousePosY = 0.0f;

// 0 - not connected; 1 - connecting; 2 - failed; 3 - connected
std::atomic<int> m_streamConnectionState = 0;

char m_ipv4Url[20] = "127.0.0.1";
int m_selectedCodec = 0;

long long unsigned int GetEpochTimeMicroSeconds()
{
	auto now = std::chrono::system_clock::now();

	// reutrn microseconds since epoch
	return now.time_since_epoch() / std::chrono::microseconds(1);
}

void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}

	std::uint8_t cameraInputBitMask = 0;
	//std::uint8_t lightInputBitMask = 0;

	if (action == GLFW_PRESS)
	{
		switch (key)
		{
		// Camera input
		case GLFW_KEY_W:
			m_movementInputBitmask |= 1 << 0;
			break;
		case GLFW_KEY_S:
			m_movementInputBitmask |= 1 << 1;
			break;
		case GLFW_KEY_A:
			m_movementInputBitmask |= 1 << 2;
			break;
		case GLFW_KEY_D:
			m_movementInputBitmask |= 1 << 3;
			break;
		case GLFW_KEY_E:
			m_movementInputBitmask |= 1 << 4;
			break;
		case GLFW_KEY_Q:
			m_movementInputBitmask |= 1 << 5;
			break;
		case GLFW_KEY_LEFT_SHIFT:
			m_movementInputBitmask |= 1 << 6;
			break;
		
		// Light input
		case GLFW_KEY_UP:
			m_lightInputBitmask |= 1 << 0;
			break;
		case GLFW_KEY_DOWN:
			m_lightInputBitmask |= 1 << 1;
			break;
		case GLFW_KEY_LEFT:
			m_lightInputBitmask |= 1 << 2;
			break;
		case GLFW_KEY_RIGHT:
			m_lightInputBitmask |= 1 << 3;
			break;
		default:
			break;
		}
	}

	if (action == GLFW_RELEASE)
	{
		switch (key)
		{
		// Camera input
		case GLFW_KEY_W:
			m_movementInputBitmask &= ~(1 << 0);
			break;
		case GLFW_KEY_S:
			m_movementInputBitmask &= ~(1 << 1);
			break;
		case GLFW_KEY_A:
			m_movementInputBitmask &= ~(1 << 2);
			break;
		case GLFW_KEY_D:
			m_movementInputBitmask &= ~(1 << 3);
			break;
		case GLFW_KEY_E:
			m_movementInputBitmask &= ~(1 << 4);
			break;
		case GLFW_KEY_Q:
			m_movementInputBitmask &= ~(1 << 5);
			break;
		case GLFW_KEY_LEFT_SHIFT:
			m_movementInputBitmask &= ~(1 << 6);
			break;

		// Light input
		case GLFW_KEY_UP:
			m_lightInputBitmask &= ~(1 << 0);
			break;
		case GLFW_KEY_DOWN:
			m_lightInputBitmask &= ~(1 << 1);
			break;
		case GLFW_KEY_LEFT:
			m_lightInputBitmask &= ~(1 << 2);
			break;
		case GLFW_KEY_RIGHT:
			m_lightInputBitmask &= ~(1 << 3);
			break;
		default:
			break;
		}
	}

}

void MouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		m_mousePosX = 0.0f;
		m_mousePosY = 0.0f;
		firstMouse = true;
		return;
	}

	// If this is the first time the callback is called, initialize the last position
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	// Update the last positions


	int wWidth = 1, wHeight = 1;
	glfwGetWindowSize(window, &wWidth, &wHeight);

	double mouseDeltaX = xpos - lastX;
	double mouseDeltaY = ypos - lastY;

	double normalizedDeltaX = mouseDeltaX / wWidth;
	double normalizedDeltaY = mouseDeltaY / wHeight;


	m_mousePosX = normalizedDeltaX;
	m_mousePosY = normalizedDeltaY;


	lastX = xpos;
	lastY = ypos;
}

std::function<void()> UICallback(SC::FFmpegDemuxer* demuxer, SC::NVDecoder* decoder)
{

	auto returnFunction = [decoder, demuxer]() {
		float maxX = ImGui::CalcTextSize("- Codec framerate:\t").x;

		std::string codec = "Codec not found";
		ImGui::SeparatorText("Codec info");
		if (!demuxer)
		{
			ImGui::Text("No demuxer found, cannot get codec info");
		}
		else
		{
			codec = demuxer->GetVideoCodecName();

			ImGui::Text("- Codec:");
			ImGui::SameLine(maxX);
			ImGui::Text(codec.c_str());

			ImGui::Text("- Framerate:");
			ImGui::SameLine(maxX);
			ImGui::Text("%d/%d", demuxer->GetFramerateNumerator(), demuxer->GetFramerateDenominator());

			int width = demuxer->GetWidth();
			int height = demuxer->GetHeight();
			int gcdVal = std::gcd(width, height);


			ImGui::Text("- Resolution:");
			ImGui::SameLine(maxX);
			ImGui::Text("%d x %d (%d:%d)", width, height, width / gcdVal, height / gcdVal);

			const char* chromaFormatName = demuxer->GetChromaFormat();
			ImGui::Text("- Chroma format:");
			ImGui::SameLine(maxX);
			ImGui::Text("%s", chromaFormatName);
		}
	};

	return returnFunction;
}

std::function<void()> UIStartScreenCallback(SC::StreamRenderer* sr, SC::FFmpegDemuxer* demuxer)
{
	auto returnFunction = [sr, demuxer]()
		{
			const char* label = "Connect to stream";
			ImVec2 itemSize = ImGui::CalcTextSize(label);

			int width = sr->GetWidth();
			int height = sr->GetHeight();

			float inputSize = width * 0.1f;

			ImGui::SetCursorPos(ImVec2((width - itemSize.x) * 0.5f, height * 0.1f));
			ImGui::Text("%s", label);

			ImVec2 inputPos = ImVec2((width - inputSize) * 0.5f, height * 0.1f);
			inputPos.y += itemSize.y + 10;

			// Determine whether the button should be disabled
			int connectionState = m_streamConnectionState.load();
			bool isConnecting = connectionState == 1;


			// Disable the button if the time passed is less than 10 seconds
			ImGui::BeginDisabled(isConnecting);
			ImGui::SetCursorPos(inputPos);
			ImGui::PushItemWidth(inputSize);
			ImGui::InputText("Stream address", m_ipv4Url, 256);

			inputPos.x = (width - inputSize) * 0.5f;
			inputPos.y += 20 + 10;

			const char* codecListboxData[] = { "HEVC", "H264" };

			ImGui::PushItemWidth(inputSize);
			ImGui::SetCursorPos(inputPos);
			ImGui::ListBox("Select a codec", &m_selectedCodec, codecListboxData, 2, 2);

			itemSize = ImGui::CalcTextSize("Connecting...");
			inputPos.x = (width - itemSize.x) * 0.5f;
			inputPos.y += (ImGui::GetItemRectMax().y - ImGui::GetItemRectMin().y) + 20;
			ImGui::SetCursorPos(inputPos);



			// Set the label for the button based on whether it's disabled or not
			std::string connectLabel = isConnecting ? "Connecting..." : "Connect";


			if (ImGui::Button(connectLabel.c_str(), ImVec2(itemSize.x + 10, 20)))
			{
				m_streamConnectionState.store(1);

				std::thread connectThread([demuxer]() {

					m_clientHost.Connect(m_ipv4Url, 2345);

					for (int i = 0; i < 5; i++)
					{
						Sleep(20);
						if (m_clientHost.IsConnected() && m_clientHost.HasPeers())
						{
							break;
						}
						else
						{
							m_streamConnectionState.store(2);
							m_clientHost.Disconnect();
							return;
						}
					}

					Commons::PacketGuard inputPacket = m_clientHost.CreatePacket();
					inputPacket->ClearPacket();

					if (m_selectedCodec == 0) {
						inputPacket->AppendToBuffer("STR265");
					}
					else
					{
						inputPacket->AppendToBuffer("STR264");
					}

					inputPacket->SetPacketType(Commons::NetworkPacket::PacketType::PACKET_RELIABLE);
					m_clientHost.SendData(inputPacket);

					std::string codecUrl = "udp://" + std::string(m_ipv4Url) + ":1234?overrun_nonfatal=1&fifo_size=50000000";
					if (demuxer->OpenStream(codecUrl.c_str()))
					{
						m_streamConnectionState.store(3);
					}
					else
					{
						m_streamConnectionState.store(2);
						m_clientHost.Disconnect();
					}

				});

				connectThread.detach();
			}

			if (connectionState == 2)
			{
				itemSize = ImGui::CalcTextSize("Could not connect to host.");
				inputPos.x = (width - itemSize.x) * 0.5f;
				inputPos.y += 20 + 10;
				
				ImGui::SetCursorPos(inputPos);
				ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Could not connect to host.");
			}

			ImGui::EndDisabled();
		};

	return returnFunction;
}

void DecodeFrame(SC::NVDecoder* decoder, SC::FFmpegDemuxer* demuxer, SC::StreamRenderer* renderer)
{
	assert(decoder != nullptr);
	assert(demuxer != nullptr);
	assert(renderer != nullptr);

	g_isDecodeDone = false;

	int width = (demuxer->GetWidth() + 1) & ~1;
	int height = demuxer->GetHeight();
	int nPitch = width * 4;


	int nVideoBytes = 0, nFrameReturned = 0, iMatrix = 0;
	uint8_t* pVideo = NULL;

	int nFrame = 0;


	float targetFramerate = 1.0f / 30.0f;

	lastTime = glfwGetTime();

	int number = 0;

	int n = 0;
	do
	{

		demuxer->Demux(&pVideo, &nVideoBytes);
		nFrameReturned = decoder->Decode(pVideo, nVideoBytes, CUVID_PKT_ENDOFPICTURE, n++);

		if (nFrameReturned && !nFrame)
		{
			SC_LOG_INFO("Number of Frames: {0}", decoder->GetVideoInfo());
			renderer->msfps = (1000.f) / (decoder->GetVideoFormat().frame_rate.numerator / (float)decoder->GetVideoFormat().frame_rate.denominator);
		}

		for (int i = 0; i < nFrameReturned; i++)
		{
			uint8_t* pFrame = decoder->GetFrame();
			SC::FrameData* data = renderer->GetDeviceFrameBuffer(&nPitch);

			if (!data)
			{
				return;
			}


			decoder->ConvertFrame(pFrame, data->devPtr, nPitch);
			renderer->PushReadFrame(data);

			
		}


		nFrame += nFrameReturned;

	} while (nVideoBytes > 0 && !renderer->isDone);
	
	
	g_isDecodeDone = true;
}


int main()
{
	SC::Logger::InitializeResources();

	CUDA_SAFE_CALL(cuInit(0));
	int deviceCount = 0;

	CUDA_SAFE_CALL(cuDeviceGetCount(&deviceCount));

	if (deviceCount <= 0)
	{
		SC_LOG_ERROR("No CUDA devices found");
		return -1;
	}

	cudaDeviceProp bestDeviceProp;
	int bestDeviceIndex = 0;

	cudaGetDeviceProperties(&bestDeviceProp, 0);

	for (int i = 1; i < deviceCount; i++)
	{
		cudaDeviceProp currentDeviceProp;
		cudaGetDeviceProperties(&currentDeviceProp, i);

		if (currentDeviceProp.multiProcessorCount > bestDeviceProp.multiProcessorCount)
		{
			bestDeviceProp = currentDeviceProp;
			bestDeviceIndex = i;
		}
	}

	CUcontext cuContext = NULL;
	CUdevice cuDevice = 0;
	CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, bestDeviceIndex));
	char szDeviceName[80];
	CUDA_SAFE_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
	SC_LOG_INFO("GPU in use: {0}", szDeviceName);
	CUDA_SAFE_CALL(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));

	Commons::NetworkHost::InitializeEnet();



	{
		SC::StreamRenderer sr(cuContext);

		if (!sr.InitializeGL())
		{
			SC_LOG_ERROR("Failed to initialize OpenGL");
			return -1;
		}

		SC::FFmpegDemuxer demuxer;

		
		sr.SetUICallback(UIStartScreenCallback(&sr, &demuxer));

		while (!sr.ShouldCloseWindow())
		{
			sr.Update();
			sr.Render(false);

			if (m_streamConnectionState.load() == 3)
			{
				break;
			}
		}


		if (sr.ShouldCloseWindow())
		{
			Commons::NetworkHost::DeinitializeEnet();
			CUDA_SAFE_CALL(cuCtxDestroy(cuContext));
			return 0;
		}

		//SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("https://cdn.radiantmediatechs.com/rmp/media/samples-for-rmp-site/04052024-lac-de-bimont/hls/playlist.m3u8");
		//SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4");

		SC::NVDecoder dec(cuContext, true, SC::FFmpegDemuxer::FFmpeg2NvCodecId(demuxer.GetVideoCodecID()), true, false, NULL, NULL, false, 0, 0, 1000, true);

		int width = (demuxer.GetWidth() + 1) & ~1;
		int height = demuxer.GetHeight();
		int nPitch = width * 4;

		CUdeviceptr devPtr;

		g_isDecodeDone = false;


		std::thread decodeThread(DecodeFrame, &dec, &demuxer, &sr);


		sr.InitializeResources(width, height);

		sr.SetKeyCallback(KeyCallback);
		sr.SetMouseCallback(MouseCallback);
		sr.SetUICallback(UICallback(&demuxer, &dec));

		lastTime = 0.0f;

		bool shouldClose = sr.ShouldCloseWindow() || (g_isDecodeDone && sr.IsReadQueueEmpty());

		unsigned int i = 0;

		double accumulatedTime = 0;

		double renderStartTime = glfwGetTime();
		double lastTime = renderStartTime;

		while (!shouldClose)
		{
			sr.isDone = shouldClose = sr.ShouldCloseWindow() || (g_isDecodeDone && sr.IsReadQueueEmpty());

			double totalTime = glfwGetTime();
			double deltaTime = totalTime - lastTime;
			lastTime = totalTime;

			accumulatedTime += deltaTime;

			sr.Update();

			if (m_clientHost.IsConnected() && m_clientHost.HasPeers())
			{
				Commons::PacketGuard inputPacket = m_clientHost.CreatePacket();
				inputPacket->ClearPacket();

				inputPacket->AppendToBuffer("STRINP");
				inputPacket->SetPacketType(Commons::NetworkPacket::PacketType::PACKET_RELIABLE);
				inputPacket->AppendToBuffer(GetEpochTimeMicroSeconds());
				inputPacket->AppendToBuffer(m_movementInputBitmask);
				inputPacket->AppendToBuffer(m_mousePosX);
				inputPacket->AppendToBuffer(m_mousePosY);
				inputPacket->AppendToBuffer(m_lightInputBitmask);
				inputPacket->AppendToBuffer((float)1.0f / 60.0f);
				m_clientHost.SendData(inputPacket);

			}
			sr.Render();

		}

		sr.FreeQueues();
		decodeThread.join();
	}


	Commons::NetworkHost::DeinitializeEnet();
	CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

	return 0;
}