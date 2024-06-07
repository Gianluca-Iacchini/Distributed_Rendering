#pragma once

#include "nvEncodeAPI.h"

namespace DX12Lib {

    struct NvEncInputFrame
    {
        void* inputPtr = nullptr;
        UINT pitch = 0;
    };

    class NVEncoder
    {
        using ENCODED_PACKET = std::vector<std::vector<std::uint8_t>>;

    public:
        NVEncoder() = default;
        ~NVEncoder();

        void Initialize(UINT width, UINT height);
        void EncodeFrame();
        void EndEncode();
        NvEncInputFrame& GetNextInputFrame();
        void SendResourceForEncode(CommandContext& context, Resource& resource);
        ENCODED_PACKET& GetEncodedPackets() { return m_encodedPackets; }

        void StartEncodeLoop();
        void StopEncodeLoop();

        std::vector<uint8_t> ConsumePacket();

        bool IsEncoding() { return !m_stopEncoding; }

    private:
        bool SupportsAsyncMode(GUID codecGUID);
        /// <summary>
        /// Allocate buffers to hold the input frames
        /// </summary>
        /// <param name="nInputBuffers"></param>
        void AllocateInputBuffers(UINT nInputBuffers);
        void AllocateOutputBuffers(UINT nOutputBuffers);
        void RegisterInputResources(UINT width, UINT height);
        void RegisterOutputResources(UINT bfrSize);
        void WaitForFence(ID3D12Fence* fence, UINT64 fenceValue);
        NV_ENC_REGISTERED_PTR RegisterResource(void* buffer, 
            int width, int height, 
            NV_ENC_BUFFER_FORMAT bufFormat, 
            NV_ENC_BUFFER_USAGE bufUsage,
            NV_ENC_FENCE_POINT_D3D12* inputFencePoint = nullptr);

        void MapResource(UINT buffIndex);

        NVENCSTATUS Encode(NV_ENC_INPUT_RESOURCE_D3D12* inputResource, NV_ENC_OUTPUT_RESOURCE_D3D12* outputResource);


        void FlushEncoder();
        void ReleaseInputBuffers();
        void ReleaseOutputBuffers();
        void UnregisterInputResources();
        void UnregisterOutputResources();

        void* GetCompletionEvent(UINT index) { return (m_completionEvents.size() == m_nEncodedBuffer) ? m_completionEvents[index] : nullptr; }

        void GetEncodedPacket(ENCODED_PACKET& packet, bool outputDelay);

        void SendEOS();
        void WaitForCompletionEvent(UINT event);

        ENCODED_PACKET m_encodedPackets;

        void EncodeThreadLoop();

    private:
        void* m_hEncoder = nullptr;
        NV_ENC_INITIALIZE_PARAMS m_initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        NV_ENC_CONFIG m_encodeConfig = { NV_ENC_CONFIG_VER };
        //GUID m_codeGUID = NV_ENC_CODEC_H264_GUID;
        GUID m_codeGUID = NV_ENC_CODEC_HEVC_GUID;
        GUID m_presetGUID = NV_ENC_PRESET_P1_GUID;
        UINT m_width = 0;
        UINT m_height = 0;

        std::thread encodeThread;
        bool m_stopEncoding = true;


        UINT m_nEncodedBuffer = 0;
        UINT m_iToSend = 0;
        int m_iGot = 0;

        std::vector<void*> m_completionEvents;

        std::vector<NV_ENC_INPUT_PTR> m_mappedInputBuffers;
        std::vector<NV_ENC_OUTPUT_PTR> m_mappedOutputBuffers;

        std::vector<std::unique_ptr<PixelBuffer>> m_availableResourceBuffer;
        std::queue<PixelBuffer*> m_inputCopyQueue;
        std::queue<PixelBuffer*> m_bufferCopyQueue;

        std::vector<std::unique_ptr<PixelBuffer>> m_inputBuffers;
        std::vector<std::unique_ptr<PixelBuffer>> m_outputBuffers;

        std::vector<std::unique_ptr<NV_ENC_INPUT_RESOURCE_D3D12>> m_inputResources;
        std::vector<std::unique_ptr<NV_ENC_OUTPUT_RESOURCE_D3D12>> m_outputResources;

        std::vector<NV_ENC_REGISTERED_PTR> m_registeredResources;
        std::vector<NV_ENC_REGISTERED_PTR> m_registeredResourcesOutputBuffers;


        Microsoft::WRL::ComPtr<ID3D12Fence> m_inputFence;
        Microsoft::WRL::ComPtr<ID3D12Fence> m_outputFence;

        std::vector<NvEncInputFrame> m_inputFrames;

        UINT64 m_inputFenceValue = 0;
        UINT64 m_outputFenceValue = 0;
        HANDLE m_fenceEvent = nullptr;

        const DXGI_FORMAT m_bufferFormat = DXGI_FORMAT_B8G8R8A8_UNORM;

        std::mutex m_encoderMutex;
        std::mutex m_networkMutex;

    public:
        UINT maxFrames = 60;

    private:
        static NV_ENCODE_API_FUNCTION_LIST m_nvEncodeAPI;



    };

    

    // From Nvidia Video Codec Samples

    class NVENCException
    {
    public:
        NVENCException(const std::string& errorStr, const NVENCSTATUS errorCode)
            : m_errorString(errorStr), m_errorCode(errorCode) {}

        virtual ~NVENCException() throw() {}
        virtual const char* what() const throw() { return m_errorString.c_str(); }
        NVENCSTATUS  getErrorCode() const { return m_errorCode; }
        const std::string& getErrorString() const { return m_errorString; }
        static NVENCException makeNVENCException(const std::string& errorStr, const NVENCSTATUS errorCode,
            const std::string& functionName, const std::string& fileName, int lineNo);
    private:
        std::string m_errorString;
        NVENCSTATUS m_errorCode;
    };

    inline NVENCException NVENCException::makeNVENCException(const std::string& errorStr, const NVENCSTATUS errorCode, const std::string& functionName,
        const std::string& fileName, int lineNo)
    {
        std::ostringstream errorLog;
        errorLog << functionName << " : " << errorStr << " at " << fileName << ":" << lineNo << std::endl;
        NVENCException exception(errorLog.str(), errorCode);
        return exception;
    }
}

#define NVENC_THROW_ERROR( errorStr, errorCode )                                                         \
    do                                                                                                   \
    {                                                                                                    \
        throw DX12Lib::NVENCException::makeNVENCException(errorStr, errorCode, __FUNCTION__, __FILE__, __LINE__); \
    } while (0)


#define NVENC_API_CALL( nvencAPI )                                                                                 \
    do                                                                                                             \
    {                                                                                                              \
        NVENCSTATUS errorCode = nvencAPI;                                                                          \
        if( errorCode != NV_ENC_SUCCESS)                                                                           \
        {                                                                                                          \
            std::ostringstream errorLog;                                                                           \
            errorLog << #nvencAPI << " Error: " << errorCode;                                              \
            DX12Lib::NVENCException error = DX12Lib::NVENCException::makeNVENCException(errorLog.str(), errorCode, __FUNCTION__, __FILE__, __LINE__); \
            DXLIB_CORE_ERROR("[NVEncode]: {0}", error.what());                                                                                                   \
            __debugbreak();                                                                                        \
        }                                                                                                          \
    } while (0)
