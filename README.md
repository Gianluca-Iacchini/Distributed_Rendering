# DX12 Real-Time Global Illumination (GI) Project

This repository contains a real-time global illumination (RTGI) implementation using DirectX 12. This project showcases a distributed rendering pipeline where the GI can be computed on a machine and be used by another. The project is structured into three subprojects:

- **ClusteredVoxelGI**: Computes global illumination using voxel clustering and Ray-tracing. The results can either be visualized locally or be sent to another machine.
- **LocalIllumination**: Renders a 3D scene using common rasterize techniques but does not compute global illumination. It can however receive GI data from the netwrok and composite it over the scene.
- **StreamingClient**: A sample streaming client application to showcase the difference between sending GI data only or streaming the whole scene over the network as a real-time video.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Building the Project](#building-the-project)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [License](#license)

## Introduction

The goal how this project is to showcases an implementation of a distributed rendering paradigm, where part of the rendering computation is offloaded to a different machine.
Popular distributed rendering techniques offload the rendering task entirely (or almost entirely) to a server and stream the result as a real-time video to the client [^1]

This project demonstrates real-time global illumination (GI) for 3D scenes using DX12. The system is designed with two primary workflows: 

1. **Server-side GI Computation**: The server performs clustered voxel-based GI calculations using the `ClusteredVoxelGI` component. This generates the GI data for the scene.
2. **Client-side GI Composition**: The client (`LocalIllumination`) receives this data over the network and composites it onto a direct illumination-only scene.

The **StreamingClient** allows real-time streaming of the scene data, enabling the client to visualize the server's GI results and interact with the scene.

## Requirements

- Visual Studio 2019 or later with DX12 SDK
- Windows 10 or later
- CMake for building the project
- NVIDIA or AMD GPU with DX12 support
- Python (for asset pipeline, if applicable)

## Building the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/DX12-RealTime-GI.git

## References
[^1]: https://www.nvidia.com/en-us/geforce-now/
