#include <iostream>
#include "StreamRenderer.h"

#include "Helpers.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "FFmpegDemuxer.h"
#include "NVDecoder.h"
#include "ColorSpace.h"





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
		SC::FFmpegDemuxer demuxer = SC::FFmpegDemuxer("C:/Users/iacco/Desktop/DistributedRendering/build_vs2022/LocalIllumination/output.h265");
		SC::NVDecoder dec(cuContext, true, SC::FFmpegDemuxer::FFmpeg2NvCodecId(demuxer.GetVideoCodecID()));

		int width = (demuxer.GetWidth() + 1) & ~1;
		int height = demuxer.GetHeight();
		int nPitch = width * 4;

		CUdeviceptr devPtr;
		int nVideoBytes = 0, nFrameReturned = 0, iMatrix = 0;
		uint8_t* pVideo = NULL;
		uint8_t* pFrame;

		SC::StreamRenderer sr(width, height);
		sr.Init(cuContext);

		int nFrame = 0;

		do
		{
			demuxer.Demux(&pVideo, &nVideoBytes);
			nFrameReturned = dec.Decode(pVideo, nVideoBytes);

			if (nFrameReturned && !nFrame)
			{
				SC_LOG_INFO("Number of Frames: {0}", dec.GetVideoInfo());
			}

			for (int i = 0; i < nFrameReturned; i++)
			{
				pFrame = dec.GetFrame();

				sr.GetDeviceFrameBuffer(&devPtr, &nPitch);
				iMatrix = dec.GetVideoFormat().video_signal_description.matrix_coefficients;



				if (dec.GetBitDepth() == 8)
				{
					if (dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
					{
						YUV444ToColor32<BGRA32>(pFrame, dec.GetWidth(), (uint8_t*)devPtr, nPitch, dec.GetWidth(), dec.GetHeight(), iMatrix);
					}
					else
					{
						Nv12ToColor32<BGRA32>(pFrame, dec.GetWidth(), (uint8_t*)devPtr, nPitch, dec.GetWidth(), dec.GetHeight(), iMatrix);
					}
				}
				else
				{
					if (dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
					{
						YUV444P16ToColor32<BGRA32>(pFrame, 2 * dec.GetWidth(), (uint8_t*)devPtr, nPitch, dec.GetWidth(), dec.GetHeight(), iMatrix);
					}
					else
					{
						P016ToColor32<BGRA32>(pFrame, 2 * dec.GetWidth(), (uint8_t*)devPtr, nPitch, dec.GetWidth(), dec.GetHeight(), iMatrix);
					}
				}

				sr.Update();
				sr.Render();
				
			}



			nFrame += nFrameReturned;

		} while (nVideoBytes > 0);



		while (!sr.ShouldCloseWindow())
		{
			sr.Update();
			sr.Render();
		}
	} 

	CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

	return 0;
}