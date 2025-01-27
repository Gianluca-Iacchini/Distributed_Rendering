# Usage

## Clustered Voxel GI
Before rendering the scene, an appropriate voxel grid size must be chosen. The available grid sizes are 64, 128, and 256. A higher grid size results in more accuracy but also requires more computational power.
Please note that a grid resolution of 64 is too small to provide accurate results, it is included to show how different grid sizes affect the computation.

An appropriate clusterization level must also be selected, with higher values resulting in fewer clusters being created. To obtain the best results, this value should be adjusted based on the voxel resolution and screen size.

Once the desired options have been chosen, the scene is rendered, and the radiance is computed. Here's a brief overview of the various sections:

### Voxelization Info
Displays information regarding the current radiance computation costs, such as time and memory usage.

### Light
Allows to change properties of both direct and indirect illumination.

* Light Color
  * Sets the light color
* Light Intensity
  * Sets the strength of the light
* Far Voxels Bounce Strength
  * Controls how much light should be reflected by distant voxels.
* Close Voxels Strength
  * Controls how much light should be reflected by nearby voxels.
* Light Update Frequency
  * Controls how often radiance computation should be dispatched.
* Light Lerp Frequency
  * Sets the maximum time for lerping from the old radiance to the new one.

### Gaussian Filter
Allows to change properties regarding the gaussian filtering step

* Kernel Size
  * Sets the kernel size used by the Gaussian smoothing (i.e., how many nearby voxels should be considered for smoothing each voxel's radiance).
* Sigma Value
  * Sets the smoothness strength
* Gaussian Pass Count
  * Shows the difference between using different pass counts, ideally two should always be used.
 
### Post Processing
Allows you to change the properties of compositing the filtered radiance to the scene.

* Spatial Sigma
  * Sets the blur strength based on the nearby fragments' world distance.
* Intensity Strength
  * Sets the blur strength based on the nearby fragments' radiance.
* Max World Position Threshold
  * Fragments with radiance whose world position exceeds this threshold will not be blurred together.
* Kernel Size
  * Sets the kernel size used for blurring (i.e., how many nearby fragments should be considered for blurring the radiance).

### Networking
Starts the CVGI server and shows networking info

* Start Server
  * Uses ENet to start a server on any of the machine avilable IPs
* Compression Level
  * Uses zstd to set the compression level. Higher values result in more compression, but they also take longer.
 
### Debugging
Shows different steps of the radiance computation

## Local Illumination
As a client, usage is similar to the one explained in the previous section.
As a video streaming server, the application automatically listens for incoming connections. If a connection is found, the encoder and video streamer are initialized.

## Streaming Client
The streaming client only needs to insert the desired Streaming Server address to start receiving video frames.

