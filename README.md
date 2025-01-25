# DX12 Real-Time Global Illumination (GI) Project

This repository contains a real-time global illumination (RTGI) implementation made with C++ and DirectX 12. It showcases a distributed rendering pipeline where global illumination can be computed on one machine and used by another. The project is structured into three subprojects:

* ClusteredVoxelGI: Computes global illumination using voxel clustering and ray-tracing. The results can be visualized locally or sent to another machine for compositing.
* LocalIllumination: Renders a 3D scene using standard rasterization techniques without computing global illumination. However, it can receive GI data from the network and composite it onto the scene.
* StreamingClient: A sample client application that demonstrates the difference between sending only GI data versus streaming the entire scene over the network as a real-time video.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Clustered-Voxel GI](#clustered-voxel-gi)
4. [Local Illumination](#local-illumination)
5. [Streaming Client](#streaming-client)
6. [Building the Project](#building-the-project)
7. [Project Structure](#project-structure)
8. [Usage](#usage)
9. [License](#license)

## Introduction

This project demonstrates a distributed rendering system using DX12, where rendering computations are offloaded to a server, and the results are streamed to the client in real-time. In traditional distributed rendering, the server handles most of the rendering workload, streaming the final image to the client [^1]. This allows users with lower-end hardware to run the application without needing specialized resources, while also simplifying development by reducing the need to optimize for various system configurations.

However, real-time streaming introduces challenges. Unlike standard video streaming, real-time rendering requires constant user input. Video buffering isn't feasible because the scene depends entirely on user actions. This means a stable internet connection with sufficient bandwidth is essential, and users may experience input lag due to the client being unaware of the scene's current state.
Moreover, most modern devices have some sort of 3D rendering capabilities; thus, by relying on the device solely as a decoder with an internet connection, we are underutilizing its potential. A better approach would be to leverage the device's capabilities to render a simplified version of the scene, while using an external server to compute additional details or augmentation. This method not only maximizes the use of the user's hardware but also helps mitigate input lag, as the simplified scene provides immediate feedback to user actions.

In this project, the rendering workload is split between the server and the client by separating lighting computations into direct and indirect (GI) lighting. The client renders the scene using standard rasterization techniques to compute direct lighting, while the server calculates the global illumination and sends it to the client. The client then composites the illumination with the direct lighting to produce the final frame. 

While this approach is not new and has been studied previously [^2], this repository is dedicated to revisiting the solution by employing a Clustered-Voxel technique to perform the global illumination calculations. 

## Clustered-Voxel GI
The clustered-Voxel GI project showcases how to compute the global illumination remotely. It is based on a clustered voxel technique [^3] to compute the indirect lightning, however it differs from the original implementation in the following ways.

* It uses Fast-SLIC [^4] instead of the original SLIC algorithm [^5], as Fast-SLIC is specifically designed for real-time applications.
* It employs Ray-Tracing to perform cluster visibility calculations.
* Several adjustments have been made to optimize the transmission of radiance data across the network:
   * Radiance is packed more aggressively to reduce the data that needs to be transmitted.
   * The radiance computation process is broken down into more distinct steps, allowing for more flexible client workload management (e.g., adjusting the amount of work assigned to the client).
   * The implementation has been rewritten to use DX12 instead of Vulkan as the graphics API.

A more detailed description of the project can be found here.

## Local Illumination
The Local Illumination project showcases two different types of distributed rendering: distributed global illumination and real-time video streaming.
In both cases, a simple demo scene is rendered. This scene does not compute global illumination but uses a small constant value for ambient lighting.

For distributed global illumination, this project functions as the client, receiving GI data from the server and using it to composite the final frame on the screen.
For real-time video streaming, this project serves as the server. It utilizes hardware acceleration to encode the rendered frame into a video and stream it to the client.

A more detailed description of the project can be found here.

## Streaming Client
The Streaming Client project showcases a client application for real-time video streaming. It uses hardware acceleration to decode the video stream and display it on the screen.

## Requirements

- Visual Studio 2017 or later with DX12 SDK
- Windows 10 or later
- CMake for building the project
- NVIDIA GPU with Ray-Tracing capabilities.

## Building the Project

1. Clone this repository
2. Launch the file `GenerateSolution.bat` to build the solution

You should find a `build_vs(your_vs_version)` folder with the solution inside.

## References
[^1]: https://www.nvidia.com/en-us/geforce-now/
[^2]: https://research.nvidia.com/publication/2021-07_distributed-decoupled-system-losslessly-streaming-dynamic-light-probes-thin
[^3]: https://www.sciencedirect.com/science/article/pii/S009784932200005X
[^4]: https://github.com/Algy/fast-slic
[^5]: https://dl.acm.org/doi/10.1109/TPAMI.2012.120
