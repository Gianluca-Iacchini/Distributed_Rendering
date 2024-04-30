#pragma once
#include "Helpers.h"

class Mesh;

class Model
{
public:
	Model() {}
	~Model() {}

	std::vector<std::shared_ptr<Mesh>> LoadFromFile(const std::wstring filename);
	std::vector<std::shared_ptr<Mesh>> LoadFromFile(const char* filename);

protected:
	std::vector<std::shared_ptr<Mesh>> m_meshes;


};

