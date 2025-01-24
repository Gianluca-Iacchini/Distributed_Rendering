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

This project demonstrates a distributed rendering system using DX12, where rendering computations are offloaded to a server, and the results are streamed to the client in real-time. In traditional distributed rendering, the server handles most of the rendering workload, streaming the final image to the client [^1]. This allows users with lower-end hardware to run the application without needing specialized resources, while also simplifying development by reducing the need to optimize for various system configurations.

However, real-time streaming introduces challenges. Unlike standard video streaming, real-time rendering requires constant user input. Video buffering isn't feasible because the scene depends entirely on user actions. This means a stable internet connection with sufficient bandwidth is essential, and users may experience input lag due to the client being unaware of the scene's current state.
Moreover, most modern devices have some sort of 3D rendering capabilities. By relying on the device solely as a decoder with an internet connection, we are underutilizing its potential. A better approach would be to leverage the device's capabilities to render a simplified version of the scene, while using an external server to compute additional details or augmentation. This method not only maximizes the use of the user's hardware but also helps mitigate input lag, as the simplified scene provides immediate feedback to user actions.


1. **Server-side GI Computation**: The server performs clustered voxel-based GI calculations using the `ClusteredVoxelGI` component. This generates the GI data for the scene.
2. **Client-side GI Composition**: The client (`LocalIllumination`) receives this data over the network and composites it onto a direct illumination-only scene.

The **StreamingClient** allows real-time streaming of the scene data, enabling the client to visualize the server's GI results and interact with the scene.

## Requirements

- Visual Studio 2017 or later with DX12 SDK
- Windows 10 or later
- CMake for building the project
- NVIDIA GPU with Ray-Tracing capabilities.

## Building the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/DX12-RealTime-GI.git

## References
[^1]: https://www.nvidia.com/en-us/geforce-now/
