#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Helpers.h"
#include "cuda_gl_interop.h"


namespace SC
{
	class Shader
	{
	public:
		Shader(const char* vertexPath, const char* fragmentPath);
		~Shader();

		void Use() { glUseProgram(m_ID); }
		void SetBool(const std::string& name, bool value) const { glUniform1i(glGetUniformLocation(m_ID, name.c_str()), (int)value); }
		void SetInt(const std::string& name, int value) const { glUniform1i(glGetUniformLocation(m_ID, name.c_str()), value); }
		void SetFloat(const std::string& name, float value) const { glUniform1f(glGetUniformLocation(m_ID, name.c_str()), value); }

	private:
		void CheckCompileErrors(unsigned int shader, std::string type);

	private:
		unsigned int m_ID;
	};

	struct FrameData
	{
		GLuint PBO;
		CUdeviceptr devPtr;
		cudaGraphicsResource_t cudaResource = NULL;
	};

	class StreamRenderer
	{
	public:
		StreamRenderer(CUcontext cuContext, int width, int height) : m_cuContext(cuContext), m_width(width), m_height(height) {}
		~StreamRenderer();

		bool Init(unsigned int maxFrames);

		FrameData* GetReadFrame();
		FrameData* GetWriteFrame();
		void PushReadFrame(FrameData* frameData);
		void PushWriteFrame(FrameData* frameData);

		bool ShouldCloseWindow() const { return glfwWindowShouldClose(m_window); }
		bool IsReadQueueEmpty()  { return m_frameQueues.GetOutputSize() <= 0; }

		FrameData* GetDeviceFrameBuffer(int* pnPitch);
		void CopyFrameToTexture();
		void DoneCopying();
		GLFWwindow* GetWindow() const { return m_window; }
		void Update();
		void Render();
		void FreeQueues();
		void Destroy();

		bool isDone = false;
		float msfps = 1000.0f / 30.0f;

	private:
		void ProcessInput();
		void BuildBuffers();
		void BuildTextures();
		void SetupCUDAInterop(unsigned int maxFrames);

	private:
		GLFWwindow* m_window = nullptr;
		int m_width = 1920;
		int m_height = 1080;

		GLuint m_VBO = 0;
		GLuint m_VAO = 0;
		GLuint m_EBO = 0;

		unsigned int m_texture = 0;

		std::unique_ptr<Shader> m_defaultShader = nullptr;


		bool doneCopying = false;
		bool doneDisplaying = false;


		std::vector<std::unique_ptr<FrameData>> m_frameData;
		DoubleQueue<FrameData*> m_frameQueues;

		CUcontext m_cuContext;


	};
}