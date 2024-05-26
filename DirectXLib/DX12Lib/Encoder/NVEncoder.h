#pragma once

#include "nvEncodeAPI.h"

namespace DX12Lib {

    class NVEncoder
    {
    public:
        NVEncoder() = default;
        ~NVEncoder() = default;

        void Initialize();
        void EncodeFrame(ID3D12Resource* pResource, ID3D12Resource* pOutputResource);

    private:
        void* m_hEncoder = nullptr;

    public:
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
