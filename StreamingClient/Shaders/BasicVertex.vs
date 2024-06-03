#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aPixCoord;

out vec2 pixCoords;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    pixCoords = aPixCoord;

}