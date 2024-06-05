#include <iostream>
#include "StreamRenderer.h"

#include "Helpers.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "FFmpegDemuxer.h"
#include "NVDecoder.h"
#include "ColorSpace.h"


float lastTime = 0.0f;
float currentTime = 0.0f;
float accumulatedTime = 0.0f;

bool g_isDecodeDone = false;

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
	SC::Logger::Init();

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
		SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("udp://localhost:1234");
		//SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4");
		//SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("C:/Users/iacco/Desktop/DistributedRendering/build_vs2022/LocalIllumination/output.h265");
		//SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4");
		SC::NVDecoder dec(cuContext, true, SC::FFmpegDemuxer::FFmpeg2NvCodecId(demuxer.GetVideoCodecID()));



		int width = (demuxer.GetWidth() + 1) & ~1;
		int height = demuxer.GetHeight();
		int nPitch = width * 4;

		CUdeviceptr devPtr;

		g_isDecodeDone = false;

		SC::StreamRenderer sr(cuContext, width, height);
		sr.Init(12);


		std::thread decodeThread(DecodeFrame, &dec, &demuxer, &sr);



		lastTime = 0.0f;

		bool shouldClose = sr.ShouldCloseWindow() || (g_isDecodeDone && sr.IsReadQueueEmpty());

		while (!shouldClose)
		{
			sr.isDone = shouldClose = sr.ShouldCloseWindow() || (g_isDecodeDone && sr.IsReadQueueEmpty());

			sr.Update();

			float currentTime = glfwGetTime();

			if (currentTime - lastTime > sr.msfps / 1000.0f)
			{
				sr.Render();
				
				lastTime = currentTime;
			}
		}

		sr.FreeQueues();
		decodeThread.join();
	} 


	CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

	return 0;
}