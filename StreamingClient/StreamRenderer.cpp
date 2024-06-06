#include "StreamRenderer.h"
#include "Helpers.h"
#include <fstream>
#include <sstream>

#define CHECK_OPENGL_ERROR \
{ \
	GLenum err = glGetError(); \
	if (err != GL_NO_ERROR) { \
		SC_LOG_ERROR("OpenGL error: {0}", err); \
		__debugbreak(); \
	} \
}

void CUDA_SAFE_CALL(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		SC_LOG_ERROR("CUDA Error: {0}", cudaGetErrorString(error));
		__debugbreak();
	}
}


SC::StreamRenderer::~StreamRenderer()
{
	glDeleteBuffers(1, &m_VBO);
	glDeleteBuffers(1, &m_EBO);
	glDeleteVertexArrays(1, &m_VAO);

	for (auto& data : m_frameData)
	{
		
		if (data->devPtr)
			CUDA_SAFE_CALL(cuMemFree(data->devPtr));

		if (data->PBO)
			glDeleteBuffers(1, &data->PBO);

		if (data->cudaResource)
			CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(data->cudaResource));
	}

	glfwTerminate();
}

bool SC::StreamRenderer::Init(unsigned int maxFrames)
{
	if (!glfwInit())
	{
		SC_LOG_ERROR("Failed to initialize GLFW");
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	m_window = glfwCreateWindow(m_width, m_height, "OpenGL", NULL, NULL);
	if (m_window == NULL)
	{
		SC_LOG_ERROR("Failed to create GLFW window");
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(m_window);


	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		SC_LOG_ERROR("Failed to initialize GLAD");
		return false;
	}

	SC_LOG_INFO("OpenGL Version: {0}", (char*)glGetString(GL_VERSION));

	glViewport(0, 0, m_width, m_height);


	BuildBuffers();
	BuildTextures();
	SetupCUDAInterop(maxFrames);

	std::string vertexPath = SOURCE_DIR;
	std::string fragmentPath = SOURCE_DIR;
	vertexPath += "/Shaders/BasicVertex.vs";
	fragmentPath += "/Shaders/BasicFragment.fs";
	m_defaultShader = std::make_unique<Shader>(vertexPath.c_str(), fragmentPath.c_str());

	doneDisplaying = true;

	CHECK_OPENGL_ERROR;

	return true;
}

SC::FrameData* SC::StreamRenderer::GetReadFrame()
{
	FrameData* data = nullptr;
	m_frameQueues.PopOutput(data);

	return data;
}

SC::FrameData* SC::StreamRenderer::GetWriteFrame()
{
	FrameData* data = nullptr;
	m_frameQueues.PopInput(data);

	return data;
}

void SC::StreamRenderer::PushReadFrame(FrameData* frameData)
{
	m_frameQueues.PushOutput(frameData);
}

void SC::StreamRenderer::PushWriteFrame(FrameData* frameData)
{
	m_frameQueues.PushInput(frameData);
}


void SC::StreamRenderer::Update()
{
	ProcessInput();
}

void SC::StreamRenderer::Render()
{

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	CopyFrameToTexture();

	glDisable(GL_DEPTH_TEST);

	m_defaultShader->Use();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE, m_texture);
	m_defaultShader->SetInt("streamTexture", 0);
	glBindVertexArray(m_VAO);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);

	glBindTexture(GL_TEXTURE_RECTANGLE, 0);

	glfwSwapBuffers(m_window);
	glfwPollEvents();
}

void SC::StreamRenderer::FreeQueues()
{
	m_frameQueues.SetDone();
}

void SC::StreamRenderer::Destroy()
{
}

SC::FrameData* SC::StreamRenderer::GetDeviceFrameBuffer(int* pnPitch)
{
	
	FrameData* frameData = GetWriteFrame();

	
	if (!frameData || frameData->devPtr == 0)
	{
		PushWriteFrame(frameData);
		return nullptr;
	}

	*pnPitch = m_width * 4;
	
	return frameData;
}

void SC::StreamRenderer::CopyFrameToTexture()
{

	if (glfwGetCurrentContext() != m_window) {
		SC_LOG_WARN("Current context is not the window context before setting.");
		glfwMakeContextCurrent(m_window);
		if (glfwGetCurrentContext() != m_window) {
			CHECK_OPENGL_ERROR;
		}
	}

	
	FrameData* data = GetReadFrame();

	if (!data)
		return;

	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &data->cudaResource, 0));
	CUdeviceptr backBufferDevPtr;
	size_t nSize = 0;
	CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&backBufferDevPtr, &nSize, data->cudaResource));

	CUDA_MEMCPY2D m = { 0 };
	m.srcMemoryType = CU_MEMORYTYPE_DEVICE;

	m.srcDevice = data->devPtr;
	m.srcPitch = m_width * 4;
	m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	m.dstDevice = backBufferDevPtr;
	m.dstPitch = nSize / m_height;
	m.WidthInBytes = m_width * 4;
	m.Height = m_height;
	CUDA_SAFE_CALL(cuMemcpy2DAsync(&m, 0));

	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &data->cudaResource, 0));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, data->PBO);
	glBindTexture(GL_TEXTURE_RECTANGLE, m_texture);

	// Check for OpenGL errors
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {

	}

	glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, m_width, m_height, GL_BGRA, GL_UNSIGNED_BYTE, 0);

	err = glGetError();
	if (err != GL_NO_ERROR) {
		SC_LOG_ERROR("OpenGL error after glTexSubImage2D: {0}", err);
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


	cudaStreamSynchronize(0);
	PushWriteFrame(data);
}

void SC::StreamRenderer::DoneCopying()
{

}

void SC::StreamRenderer::ProcessInput()
{
	if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(m_window, true);
	}

}

void SC::StreamRenderer::BuildBuffers()
{
	float vertices[] = {
		// positions	// Pixel Coords
	 1.0f,  1.0, 0.0f,		m_width, 0,	 // top right
	 1.0, -1.0f, 0.0f,		m_width, m_height,			 // bottom right
	-1.0, -1.0f, 0.0f,		0.0f, m_height,				 // bottom left
	-1.0,  1.0f, 0.0f,		0.0f, 0.0f,		 // top left 
	};
	unsigned int indices[] = {  // note that we start from 0!
		0, 1, 3,  // first Triangle
		1, 2, 3   // second Triangle
	};

	glGenVertexArrays(1, &m_VAO);
	glGenBuffers(1, &m_VBO);
	glGenBuffers(1, &m_EBO);
	glBindVertexArray(m_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void SC::StreamRenderer::BuildTextures()
{
	glGenTextures(1, &m_texture);
	glBindTexture(GL_TEXTURE_RECTANGLE, m_texture);
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	//glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_RECTANGLE, 0);
}

void SC::StreamRenderer::SetupCUDAInterop(unsigned int maxFrames)
{
	CUDA_SAFE_CALL(cuCtxSetCurrent(m_cuContext));


	for (unsigned int i = 0; i < maxFrames; i++)
	{
		FrameData data;
		glGenBuffers(1, &data.PBO);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, data.PBO);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 4, NULL, GL_STREAM_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		CUDA_SAFE_CALL(cuMemAlloc(&data.devPtr, m_width * m_height * 4));
		CUDA_SAFE_CALL(cuMemsetD8(data.devPtr, 0, m_width * m_height * 4));

		CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&data.cudaResource, data.PBO, cudaGraphicsRegisterFlagsWriteDiscard));

		m_frameData.push_back(std::make_unique<FrameData>(data));
		
		m_frameQueues.PushInput(m_frameData.back().get());
	}


}

SC::Shader::Shader(const char* vertexPath, const char* fragmentPath)
{
	std::string vertexCode;
	std::string fragmentCode;

	std::ifstream vShaderFile;
	std::ifstream fShaderFile;

	SC_LOG_INFO("Loading vertex shader: {0}", vertexPath);
	SC_LOG_INFO("Loading fragment shader: {0}", fragmentPath);

	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	try
	{
		vShaderFile.open(vertexPath);
		fShaderFile.open(fragmentPath);

		std::stringstream vShaderStream, fShaderStream;

		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();

		vShaderFile.close();
		fShaderFile.close();

		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
	}
	catch (std::ifstream::failure e)
	{
		SC_LOG_ERROR("Shader compilation read error: {0}", e.what());
	}

	const char* vShaderCode = vertexCode.c_str();
	const char* fShaderCode = fragmentCode.c_str();

	unsigned int vertex, fragment;
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	CheckCompileErrors(vertex, "VERTEX");

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	CheckCompileErrors(fragment, "FRAGMENT");

	m_ID = glCreateProgram();
	glAttachShader(m_ID, vertex);
	glAttachShader(m_ID, fragment);
	glLinkProgram(m_ID);
	CheckCompileErrors(m_ID, "PROGRAM");

	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

SC::Shader::~Shader()
{
	glDeleteProgram(m_ID);
}

void SC::Shader::CheckCompileErrors(unsigned int shader, std::string type)
{
	int success;
	char infoLog[1024];

	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			SC_LOG_ERROR("Shader compilation error of type {0}: {1}", type, infoLog);
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			SC_LOG_ERROR("Shader linking error of type {0}: {1}", type, infoLog);
		}
	}	
}


