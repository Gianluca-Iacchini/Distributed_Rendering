#version 330 core


uniform sampler2DRect streamTexture;

in vec2 pixCoords;
out vec4 FragColor;

void main()
{
    FragColor = texture(streamTexture, pixCoords);
}