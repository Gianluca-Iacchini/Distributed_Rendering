#include <iostream>
#include "StreamRenderer.h"

#include "Helpers.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "FFmpegDemuxer.h"
#include "NVDecoder.h"
#include "ColorSpace.h"
#include <winsock2.h>
#include <ws2tcpip.h>

float lastTime = 0.0f;
float currentTime = 0.0f;
float accumulatedTime = 0.0f;

bool g_isDecodeDone = false;


double lastX = 0.0;
double lastY = 0.0;
std::string lastMouseXTruncated = "0";
std::string lastMouseYTruncated = "0";
bool firstMouse = true;


SOCKET g_sockfd;
struct sockaddr_in g_servAddr;

int cameraForwardValue = 0;
int cameraStrafeValue = 0;
int cameraLiftValue = 0;

void InitializeWinsock()
{
	WSADATA wsaData;

	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
		SC_LOG_ERROR("WSAStartup failed.");
		return;
		return;
	}

	g_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	if (g_sockfd == INVALID_SOCKET) {
		SC_LOG_ERROR("Socket creation failed.");
		WSACleanup();
		return;
	}

	memset(&g_servAddr, 0, sizeof(g_servAddr));

	g_servAddr.sin_family = AF_INET;
	g_servAddr.sin_port = htons(12345);
	inet_pton(AF_INET, "127.0.0.1", &g_servAddr.sin_addr);

	SC_LOG_INFO("Winsock initialized");
}

void SendInput(std::string input)
{

	std::string message = input;
	if (sendto(g_sockfd, message.c_str(), message.size(), 0, (struct sockaddr*)&g_servAddr, sizeof(g_servAddr)) == SOCKET_ERROR)
	{
		int error = WSAGetLastError();
		SC_LOG_ERROR("Failed to send message {0}", error);
		return;
	}
}

void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	std::string keyInput = "";

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}

	if (action == GLFW_PRESS)
	{
		switch (key)
		{
		case GLFW_KEY_W:
			keyInput += "CF 1\n";
			break;
		case GLFW_KEY_S:
			keyInput += "CF -1\n";
			break;
		case GLFW_KEY_A:
			keyInput += "CS -1\n";
			break;
		case GLFW_KEY_D:
			keyInput += "CS 1\n";
			break;
		case GLFW_KEY_E:
			keyInput += "CL 1\n";
			break;
		case GLFW_KEY_Q:
			keyInput += "CL -1\n";
			break;
		default:
			break;
		}
	}

	if (action == GLFW_RELEASE)
	{
		if (key == GLFW_KEY_W || key == GLFW_KEY_S)
		{
			keyInput += "CF 0\n";
		}
		if (key == GLFW_KEY_A || key == GLFW_KEY_D)
		{
			keyInput += "CS 0\n";
		}
		if (key == GLFW_KEY_E || key == GLFW_KEY_Q)
		{
			keyInput += "CL 0\n";
		}
	}

	if (keyInput.empty())
	{
		return;
	}


	SendInput(keyInput);
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

	// Calculate the delta positions
	double deltaX = xpos - lastX;
	double deltaY = ypos - lastY;

	// Update the last positions
	lastX = xpos;
	lastY = ypos;

	int wWidth = 1, wHeight = 1;
	glfwGetWindowSize(window, &wWidth, &wHeight);

	double normalizedDeltaX = deltaX / wWidth;
	double normalizedDeltaY = deltaY / wHeight;

	std::string input = "M ";

	std::string truncatedX = SC::Helpers::TruncateToString(normalizedDeltaX, 5);
	std::string truncatedY = SC::Helpers::TruncateToString(normalizedDeltaY, 5);

	if (truncatedX == lastMouseXTruncated && truncatedY == lastMouseYTruncated)
		return;

	if (truncatedX != lastMouseXTruncated)
	{
		input += "x:" + truncatedX + " ";
		lastMouseXTruncated = truncatedX;
	}

	if (truncatedY != lastMouseYTruncated)
	{
		input += "y:" + truncatedY;
		lastMouseYTruncated = truncatedY;
	}

	input += "\n";

	SendInput(input);
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


	{
		SC::StreamRenderer sr(cuContext);

		if (!sr.InitializeGL())
		{
			SC_LOG_ERROR("Failed to initialize OpenGL");
			return -1;
		}

		//SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("https://cdn.radiantmediatechs.com/rmp/media/samples-for-rmp-site/04052024-lac-de-bimont/hls/playlist.m3u8");
		SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("udp://localhost:1234?overrun_nonfatal=1&fifo_size=50000000");
		//SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4");
		SC::NVDecoder dec(cuContext, true, SC::FFmpegDemuxer::FFmpeg2NvCodecId(demuxer.GetVideoCodecID()), true, false, NULL, NULL, false, 0, 0, 1000, false);


		InitializeWinsock();

		int width = (demuxer.GetWidth() + 1) & ~1;
		int height = demuxer.GetHeight();
		int nPitch = width * 4;

		CUdeviceptr devPtr;

		g_isDecodeDone = false;



		sr.InitializeResources(width, height);
		sr.SetKeyCallback(KeyCallback);
		sr.SetMouseCallback(MouseCallback);


		std::thread decodeThread(DecodeFrame, &dec, &demuxer, &sr);



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

			if (renderStartTime > 0.0f )
			{
				sr.Update();
				sr.Render();
				renderStartTime -= deltaTime;
				continue;
			}



			accumulatedTime += deltaTime;

			if ((accumulatedTime >= sr.msfps / (1000.0f)))
			{
				accumulatedTime -= sr.msfps / (1000.0f);
				sr.Update();
				sr.Render();
			}

		}

		sr.FreeQueues();
		decodeThread.join();
	} 


	CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

	return 0;
}