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

void DecodeFrame(SC::NVDecoder* decoder, SC::FFmpegDemuxer* demuxer, SC::StreamRenderer* renderer)
{
	assert(decoder != nullptr);
	assert(demuxer != nullptr);
	assert(renderer != nullptr);

	int width = (demuxer->GetWidth() + 1) & ~1;
	int height = demuxer->GetHeight();
	int nPitch = width * 4;

	CUdeviceptr devPtr;
	int nVideoBytes = 0, nFrameReturned = 0, iMatrix = 0;
	uint8_t* pVideo = NULL;
	uint8_t* pFrame;

	int nFrame = 0;

	bool shouldExit = false;

	float targetFramerate = 1.0f / 30.0f;

	lastTime = glfwGetTime();

	do
	{
		demuxer->Demux(&pVideo, &nVideoBytes);
		nFrameReturned = decoder->Decode(pVideo, nVideoBytes);

		if (nFrameReturned && !nFrame)
		{
			SC_LOG_INFO("Number of Frames: {0}", decoder->GetVideoInfo());
			targetFramerate = (1.f) / (decoder->GetVideoFormat().frame_rate.numerator / (float)decoder->GetVideoFormat().frame_rate.denominator);
		}

		for (int i = 0; i < nFrameReturned && !shouldExit; i++)
		{

			pFrame = decoder->GetFrame();

			renderer->GetDeviceFrameBuffer(&devPtr, &nPitch);
			iMatrix = decoder->GetVideoFormat().video_signal_description.matrix_coefficients;



			if (decoder->GetBitDepth() == 8)
			{
				if (decoder->GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
				{
					YUV444ToColor32<BGRA32>(pFrame, decoder->GetWidth(), (uint8_t*)devPtr, nPitch, decoder->GetWidth(), decoder->GetHeight(), iMatrix);
				}
				else
				{
					Nv12ToColor32<BGRA32>(pFrame, decoder->GetWidth(), (uint8_t*)devPtr, nPitch, decoder->GetWidth(), decoder->GetHeight(), iMatrix);
				}
			}
			else
			{
				if (decoder->GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
				{
					YUV444P16ToColor32<BGRA32>(pFrame, 2 * decoder->GetWidth(), (uint8_t*)devPtr, nPitch, decoder->GetWidth(), decoder->GetHeight(), iMatrix);
				}
				else
				{
					P016ToColor32<BGRA32>(pFrame, 2 * decoder->GetWidth(), (uint8_t*)devPtr, nPitch, decoder->GetWidth(), decoder->GetHeight(), iMatrix);
				}
			}



			while ((accumulatedTime < targetFramerate) && !shouldExit)
			{
				float currentTime = glfwGetTime();
				float deltaTime = currentTime - lastTime;


				
				renderer->Update();
				renderer->Render();

				accumulatedTime += deltaTime;
				lastTime = currentTime;
				shouldExit = renderer->ShouldCloseWindow();
			}


			// Reset accumulated time for the next frame
			accumulatedTime = 0;
			lastTime = glfwGetTime();

		}



		nFrame += nFrameReturned;

	} while (nVideoBytes > 0 && !shouldExit);
	
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
		SC::NVDecoder dec(cuContext, true, SC::FFmpegDemuxer::FFmpeg2NvCodecId(demuxer.GetVideoCodecID()));



		int width = (demuxer.GetWidth() + 1) & ~1;
		int height = demuxer.GetHeight();

		SC::StreamRenderer sr(width, height);
		sr.Init(cuContext);
		DecodeFrame(&dec, &demuxer, &sr);




	} 


	CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

	return 0;
}