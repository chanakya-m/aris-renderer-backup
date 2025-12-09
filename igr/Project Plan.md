Project Plan

1. Data Preparation
    - get a mesh of the Stanford Bunny
    - write a script using Open3D to sample points and normals from the surface and normalize everything to fit inside a unit sphere. This creates the ground truth point cloud.

2. Baseline IGR Implementation
    - Build the neural network architecture based on the IGR paper

3. Training the Baseline
    - Write the training loop from scratch
    - Two losses: the Manifold loss (forcing network output to 0 on surface points) and the Eikonal loss (forcing the gradient norm to 1 on off-surface points)
    - use the specific sampling strategy of mixing near-surface points with global random points to stabilize training.
    - Train this in full precision (FP32) until I get a clean mesh extraction using marching cubes

4. Quantization Aware Training (QAT)
    - Once the baseline works, modify the network definition to include PyTorch quantization stubs
    - I will update the training script to use the quantization-aware training workflow
        - fuse layers (Linear + Activation), prepare the model for QAT, and then fine-tune the pre-trained model for a few epochs. This simulates the lower precision during the forward pass so the weights adapt.

5. Model Conversion and Extraction
    - Convert the QAT model to actual INT8 weights.
    - Run my reconstruction script on both the FP32 model and the INT8 model to generate two resulting .obj files.

7. Evaluation and Benchmarking
Write a script to measure:
    - Geometric quality
       - sample points from my reconstructed meshes and calculate the Chamfer Distance against the original ground truth.
    - Efficiency
        - the inference latency (queries per second) and the file size on disk for both models

### Report Images and Tables
1. Visuals
    - Image showing the input point cloud.
    - Image showing the initial "sphere" state of the network before training.
    - Side-by-side render: Ground Truth Mesh vs. FP32 Reconstruction vs. INT8 Reconstruction.
    - Close-up render zooming in on high-frequency details (like the bunny ears or fur) to show if quantization caused smoothing artifacts.
    - A plot of the Training Loss curve (Manifold loss vs Eikonal loss) over time.

2. Tables
    - Main Comparison Table:
    Columns: Model Type (FP32 vs INT8), Model Size (MB), Inference Time (ms per batch), Chamfer Distance (Lower is better).
    - Hyperparameter Table:
    Listing learning rate, lambda weight for Eikonal loss, batch size, and sigma values used for sampling.



## Also:
- speed
- use a newer ML technique, compare quantization with IGR
- use different meshes than just the bunny
- use even lower quantization
- use a sparser pointcloud
