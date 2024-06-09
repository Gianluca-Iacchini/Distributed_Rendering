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
		StreamRenderer(CUcontext cuContext) : m_cuContext(cuContext), m_width(1920), m_height(1080), m_mouseCallback(MouseCallback) {}
		~StreamRenderer();

		bool InitializeGL();
		void InitializeResources(int width, int height, unsigned int maxFrames=30);

		FrameData* GetReadFrame();
		FrameData* GetWriteFrame();
		void PushReadFrame(FrameData* frameData);
		void PushWriteFrame(FrameData* frameData);
		void SetKeyCallback(GLFWkeyfun keyCallback) { glfwSetKeyCallback(m_window, keyCallback); }
		void SetMouseCallback(GLFWcursorposfun mouseCallback) { m_mouseCallback = mouseCallback; }

		bool ShouldCloseWindow() const { return glfwWindowShouldClose(m_window); }
		bool IsReadQueueEmpty()  { return m_frameQueues.GetOutputSize() <= 0; }

		FrameData* GetDeviceFrameBuffer(int* pnPitch);
		void CopyFrameToTexture();
		GLFWwindow* GetWindow() const { return m_window; }
		void Update();
		void Render();
		void FreeQueues();
		void Destroy();

		float msfps = 1000.0f / 30.0f;
		bool isDone = false;

	private:
		void BuildBuffers();
		void BuildTextures();
		void SetupCUDAInterop(unsigned int maxFrames);

		static void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void MouseCallback(GLFWwindow* window, double xpos, double ypos);

	private:
		GLFWwindow* m_window = nullptr;
		int m_width = 1920;
		int m_height = 1080;

		GLuint m_VBO = 0;
		GLuint m_VAO = 0;
		GLuint m_EBO = 0;

		unsigned int m_texture = 0;

		std::unique_ptr<Shader> m_defaultShader = nullptr;



		std::vector<std::unique_ptr<FrameData>> m_frameData;
		DoubleQueue<FrameData*> m_frameQueues;

		CUcontext m_cuContext;

		bool firstTimeStopped = true;
		float m_lastMouseX = 0;
		float m_lastMouseY = 0;

		GLFWcursorposfun m_mouseCallback = nullptr;
	};
}