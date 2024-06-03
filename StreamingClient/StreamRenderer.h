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

	class StreamRenderer
	{
	public:
		StreamRenderer(int width, int height) : m_width(width), m_height(height) {}
		~StreamRenderer();

		bool Init(CUcontext cudaContext);
		void Update();
		void Render();

		bool ShouldCloseWindow() const { return glfwWindowShouldClose(m_window); }
		void GetDeviceFrameBuffer(CUdeviceptr* framePtr, int* pnPitch);

	private:
		void ProcessInput();
		void BuildBuffers();
		void BuildTextures();
		void SetupCUDAInterop(CUcontext cudaContext);

	private:
		GLFWwindow* m_window = nullptr;
		int m_width = 1920;
		int m_height = 1080;

		CUdeviceptr m_devPtrFrame = 0;

		GLuint m_VBO = 0;
		GLuint m_VAO = 0;
		GLuint m_EBO = 0;

		// Pixel Buffer Object for CUDA-OpenGL interop
		GLuint m_pbo = 0;

		cudaGraphicsResource_t m_cudaResource = NULL;

		unsigned int m_texture = 0;

		std::unique_ptr<Shader> m_defaultShader = nullptr;
	};
}