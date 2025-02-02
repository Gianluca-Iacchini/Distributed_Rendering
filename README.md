# DX12 Real-Time Global Illumination (GI) Project

This repository contains a real-time global illumination (RTGI) implementation made with C++ and DirectX 12. It showcases a distributed rendering pipeline where global illumination can be computed on one machine and used by another.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Clustered-Voxel GI](#clustered-voxel-gi)
4. [Local Illumination](#local-illumination)
5. [Streaming Client](#streaming-client)
6. [Building the Project](#building-the-project)
7. [Project Structure](#project-structure)
8. [Usage](#usage)
9. [Results and Performance](#results-and-performance)
10. [Known Issues](#known-issues)
11. [Credits](#credits)

| ![](Images/screenshot_final.jpg) | ![](Images/screenshot_top.jpg)|
|:-------------------------------------------------:|:---------------------------------------------------------:|
| ![](Images/screenshot_blue.jpg) | ![](Images/screenshot_banners.jpg)|




## Introduction

This project demonstrates a distributed rendering system using DX12, where rendering computations are offloaded to a server, and the results are streamed to the client in real-time. In traditional distributed rendering, the server handles most of the rendering workload, streaming the final image to the client [^1]. This allows users with lower-end hardware to run the application without needing specialized resources, while also simplifying development by reducing the need to optimize for various system configurations.

However, real-time streaming introduces challenges. Unlike standard video streaming, real-time rendering requires constant user input. Video buffering isn't feasible because the scene depends entirely on user actions. This means a stable internet connection with sufficient bandwidth is essential, and users may experience input lag due to the client being unaware of the scene's current state.
Moreover, most modern devices have some sort of 3D rendering capabilities; thus, by relying on the device solely as a decoder with an internet connection, we are underutilizing its potential. A better approach would be to leverage the device's capabilities to render a simplified version of the scene, while using an external server to compute additional details or augmentation. This method not only maximizes the use of the user's hardware but also helps mitigate input lag, as the simplified scene provides immediate feedback to user actions.

In this project, the rendering workload is split between the server and the client by separating lighting computations into direct and indirect (GI) lighting. The client renders the scene using standard rasterization techniques to compute direct lighting, while the server calculates the global illumination and sends it to the client. The client then composites the illumination with the direct lighting to produce the final frame. 

While this approach is not new and has been studied previously [^2], this repository is dedicated to revisiting the solution by employing a Clustered-Voxel technique to perform the global illumination calculations. 

## Clustered-Voxel GI
The clustered-Voxel GI project showcases how to compute the global illumination remotely. It is based on a clustered voxel technique [^3] to compute the indirect lightning. The results can be displayed locally, or sent remotely to another machine.

A more detailed description of the project can be found [here](ClusteredVoxelGI/README.md).

| ![](Images/screenshot_voxel.jpg) | ![](Images/screenshot_cluster.jpg)|
|:-------------------------------------------------:|:---------------------------------------------------------:|
| ![](Images/screenshot_litVoxels.jpg) | ![](Images/screenshot_rawRadiance.jpg)|


## Local Illumination
The Local Illumination project showcases two different types of distributed rendering: distributed global illumination and real-time video streaming.
In both cases, a simple demo scene is rendered. This scene does not compute global illumination but uses a small constant value for ambient lighting.

For distributed global illumination, this project functions as the client, receiving GI data from the server and using it to composite the final frame on the screen.
For real-time video streaming, this project serves as the server. It utilizes hardware acceleration to encode the rendered frame into a video and stream it to the client.

A more detailed description of the project can be found [here](LocalIllumination/README.md).

| ![](Images/screenshot_directOnly.jpg) Ambient only | ![](Images/screenshot_networkGaussian.jpg) Network Radiance|
|:-------------------------------------------------:|:---------------------------------------------------------:|

## Streaming Client
The Streaming Client project showcases a client application for real-time video streaming. It uses hardware acceleration to decode the video stream and display it on the screen.

A more detailed description of the project can be found [here](StreamingClient/README.md).

## Requirements

- Visual Studio 2017 or later with DX12 SDK
- Windows 10 or later
- CMake for building the project
- NVIDIA GPU with Ray-Tracing capabilities
- Cuda Toolkit

## Building the Project

1. Clone this repository
2. Launch the file `GenerateSolution.bat` to build the solution

You should find a `build_vs(your_vs_version)` folder with the solution inside.

## Project Structure

 The project is structured into three subprojects:

* ClusteredVoxelGI: Computes global illumination using voxel clustering and ray-tracing. The results can be visualized locally or sent to another machine for compositing.
* LocalIllumination: Renders a basic 3D scene using commmon rasterization techniques. It can receive radiance data from another machine and use it to composite a final frame to screen. It can also act as a server for real-time video streaming.
* StreamingClient: An implementation of a simple real-time video streaming.

## Results and Performance
The project was tested on the following hardware:
- **Server**: Intel i9-12900K, NVIDIA GeForce RTX 3090, 32GB DDR5 RAM
- **Client**: Intel i7-4900, NVIDIA GeForce GTX 980, 16GB DDR3 RAM

These systems were tested under the following conditions:

1. Real Network Test
   * The client was connected to a 5GHz Wi-Fi network with a maximum available bandwidth of 50mbps.
   * The server was connected via Ethernet to a fiber-optic connection, with a maximum available bandwidth of 80mbps.

In this setup, the average round-trip time (RTT) ranged from 80ms to 100ms, with packet loss averaging between 3% and 5%.

2. Local Network
 * Both the server and client were connected to the same local network but on different subnets.

For this test, various network conditions were simulated using [clumsy](https://jagt.github.io/clumsy/) to analyze how different network issues affected performance.

### Rendering Performance
The performance of the GI algorithm is influenced by several factors, including the voxel resolution, clusterization level, light update frequency, and the number of visible voxels.
On both machines the application was rendered with a resolution of 1980x1080.

#### Radiance Server
The server's performance is influenced by the lighting conditions. When radiance is not being computed, the server application typically runs at an average of 288 fps.

* When the light changes (e.g., position, rotation, color, intensity, or radiance bounce strength), the radiance must be recomputed for all visible voxel faces.
  * For a voxel grid of 256x256x256 and a clusterization level of 1, this results in a performance drop to around 240 fps.
* When the light remains unchanged and the camera moves, the radiance needs to be recomputed only for the voxel faces that were not visible to the camera since the last light update.
  * The performance impact in this case depends on how many new voxels enter the camera's frustum. Typically, this doesn't cause a significant performance drop.
  * Voxel radiance is cached in a buffer as long as the light settings stay the same. If the camera revisits a voxel it has already seen, no new radiance computation is performed for that voxel.

#### Radiance Client
The client's performance is influenced by the same factors as the server, but to a much lesser extent. Unlike the server, the client doesn't compute the radiance; it only performs the gaussian filtering pass.
On average, the client runs at 144 fps, with a drop to around 110 fps when the light changes and a significant number of voxels are visible. When the light remains unchanged, the performance impact is negligible.

#### Streaming Server
The streaming server was tested on both machines, yielding similar results. Frame encoding and streaming caused a drop in performance, from 288 fps and 145 fps to approximately 280 fps and 125 fps, respectively.

#### Streaming Client
The streaming client was also tested on both machines. Its frame rate was capped at a stable 60 fps, which matches the encoding frame rate.

### Network Performance
#### Radiance
The network performance depends largely on the network conditions, with packet loss having the greatest impact.

In the real-world network test, significant lighting changes (such as sudden color shifts) caused noticeable delays when using a voxel grid size of 128x128x128. However, smaller lighting changes and camera movements made remote radiance computation much less noticeable.

In the local network test, the following conditions were tested:
* RTT values ranging from 1ms to 200ms
* Packet loss values ranging from 0.0% to 5.0%
* Duplicate and/or out-of-order packets
* Bandwidth limits

The application was found to be resilient to latency up to around 180ms, after which delays became noticeable for lighting changes. It was also particularly sensitive to packet loss, with delays becoming noticeable above 0.5%. Duplicate and out-of-order packets did not significantly affect the results.

Finally, the bandwidth limit's impact depended on the voxel grid size. Delays became noticeable when the bandwidth was under 2 Mbps for a 64x64x64 voxel grid and under 5 Mbps for a 128x128x128 voxel grid.

#### Streaming
Real-time video streaming was tested under similar conditions. In the real network test, small visible artifacts were observed.

Under the altered local network conditions, the streaming application was slightly less resilient to latency. An RTT of around 100ms caused noticeable delays in input feedback. However, the application was more resilient to packet loss, with 3% packet loss resulting in very visible artifacts but still allowing the application to remain usable.

## Usage
Information regarding the projects usage can be found [here](USAGE.md).

## Known Issues
Some issues remain unresolved, as addressing them would increase the complexity of the application. The main goal of this project is to showcase a distributed rendering system in a simple and clear manner.

### Lightning
Ideal Lightning configuration are dependent on voxel resolution and clusterization level

### Networking
* The application requires minimal packet loss (<0.5%).
* Repeatedly connecting and disconnecting from a server may require application restart

## Credits
[Microsoft Mini-Engine](https://github.com/microsoft/DirectX-Graphics-Samples/tree/master/MiniEngine) and [Introduction to 3D Game Programming With DirectX 12](https://www.d3dcoder.net/d3d12.htm) were used as a reference for most of the DX12 library files.

[Clustered voxel real-time global illumination](https://www.sciencedirect.com/science/article/pii/S009784932200005X) was used as the basis of the GI algorithm.

[NVIDIA Video Codec SKD](https://developer.nvidia.com/video-codec-sdk) was used as a reference for the video encoder / decoder files.

[FFmpeg](https://www.ffmpeg.org/) for video streaming

[zstd](https://github.com/facebook/zstd) for performing compression

[ENet](http://enet.bespin.org/index.html) for the network files.

[Dear imGUI](https://github.com/ocornut/imgui) for the GUI


## References
[^1]: https://www.nvidia.com/en-us/geforce-now/
[^2]: https://research.nvidia.com/publication/2021-07_distributed-decoupled-system-losslessly-streaming-dynamic-light-probes-thin
[^3]: https://www.sciencedirect.com/science/article/pii/S009784932200005X
[^4]: https://github.com/Algy/fast-slic
[^5]: https://dl.acm.org/doi/10.1109/TPAMI.2012.120

---



https://github.com/user-attachments/assets/59d5bfa6-5993-4b41-b681-2966d2cd12f2

https://github.com/user-attachments/assets/8acb3a7f-1877-4195-a227-5454f42b3370

