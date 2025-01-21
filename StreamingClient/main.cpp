#include <iostream>
#include "StreamRenderer.h"

#include "Helpers.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "FFmpegDemuxer.h"
#include "NVDecoder.h"
#include "ColorSpace.h"

#include "NetworkManager.h"

#define IS_STREAMING 0

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

float m_mousePosXY[2];

bool m_isInputDataReady = true;

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
		default:
			break;
		}
	}

	if (action == GLFW_RELEASE)
	{
		switch (key)
		{
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
		default:
			break;
		}
	}

	m_isInputDataReady = true;
}

void MouseCallback(GLFWwindow* window, double xpos, double ypos)
{
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
	double mouseDeltaY = lastY - ypos;

	double normalizedDeltaX = mouseDeltaX / wWidth;
	double normalizedDeltaY = mouseDeltaY / wHeight;


	m_mousePosXY[0] = normalizedDeltaX;
	m_mousePosXY[1] = normalizedDeltaY;

	m_isInputDataReady = true;

	lastX = xpos;
	lastY = ypos;
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


	do
	{

		demuxer->Demux(&pVideo, &nVideoBytes);
		nFrameReturned = decoder->Decode(pVideo, nVideoBytes);


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

	m_clientHost.Connect("127.0.0.1", 2345);

	{
		SC::StreamRenderer sr(cuContext);

		if (!sr.InitializeGL())
		{
			SC_LOG_ERROR("Failed to initialize OpenGL");
			return -1;
		}

#if IS_STREAMING
		//SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("https://cdn.radiantmediatechs.com/rmp/media/samples-for-rmp-site/04052024-lac-de-bimont/hls/playlist.m3u8");
		SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("udp://localhost:1234?overrun_nonfatal=1&fifo_size=50000000");
		//SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4");
		SC::NVDecoder dec(cuContext, true, SC::FFmpegDemuxer::FFmpeg2NvCodecId(demuxer.GetVideoCodecID()), true, false, NULL, NULL, false, 0, 0, 1000, false);

		int width = (demuxer.GetWidth() + 1) & ~1;
		int height = demuxer.GetHeight();
		int nPitch = width * 4;

		CUdeviceptr devPtr;

		g_isDecodeDone = false;


		std::thread decodeThread(DecodeFrame, &dec, &demuxer, &sr);
		sr.InitializeResources(width, height);
#endif


		sr.SetKeyCallback(KeyCallback);
		sr.SetMouseCallback(MouseCallback);

		lastTime = 0.0f;

		bool shouldClose = sr.ShouldCloseWindow() || (g_isDecodeDone && sr.IsReadQueueEmpty());

		unsigned int i = 0;

		double accumulatedTime = 0;

		double renderStartTime = glfwGetTime();
		double lastTime = renderStartTime;

#if IS_STREAMING
		while (!shouldClose)
		{
			sr.isDone = shouldClose = sr.ShouldCloseWindow() || (g_isDecodeDone && sr.IsReadQueueEmpty());

			double totalTime = glfwGetTime();
			double deltaTime = totalTime - lastTime;
			lastTime = totalTime;

			if (renderStartTime > 0.0f )
			{
				sr.Update();
				sr.Render(IS_STREAMING);
				renderStartTime -= deltaTime;
				continue;
			}



			accumulatedTime += deltaTime;

			if ((accumulatedTime >= sr.msfps / (1000.0f)))
			{
				accumulatedTime -= sr.msfps / (1000.0f);
				sr.Update();
				sr.Render(IS_STREAMING);
			}

		}

		sr.FreeQueues();
		decodeThread.join();
	
#else
		while (!shouldClose)
		{
			sr.isDone = shouldClose = sr.ShouldCloseWindow();
			sr.Update();

			if (m_isInputDataReady)
			{
				Commons::PacketGuard inputPacket = m_clientHost.CreatePacket();
				inputPacket->ClearPacket();
				inputPacket->AppendToBuffer("STRINP");
				inputPacket->SetPacketType(Commons::NetworkPacket::PacketType::PACKET_RELIABLE);
				inputPacket->AppendToBuffer(Commons::NetworkHost::GetEpochTime());
				inputPacket->AppendToBuffer(m_movementInputBitmask);
				inputPacket->AppendToBuffer(m_mousePosXY[0]);
				inputPacket->AppendToBuffer(m_mousePosXY[1]);
				m_clientHost.SendData(inputPacket);

				m_isInputDataReady = false;
			}

			sr.Render(IS_STREAMING);
		}

#endif
	}


	Commons::NetworkHost::DeinitializeEnet();
	CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

	return 0;
}