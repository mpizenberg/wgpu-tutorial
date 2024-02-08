# wgpu tutorial

This repository holds the simplest examples I could come up with to introduce wgpu.
The general thinking of these examples consist in avoiding unneeded complexity as much as possible.
As a result, none of the examples deal with windowing and events.
Instead, we just transfer the shader outputs into image buffers and write them directly to disk as image files.
To run an example, simply move into that directory and run its main program.

```sh
# move into the first example
cd example_01_triangle
# run the example
cargo run
# the output is written to "image.png"
```

Currently, the examples are the following.

## 1. Setup and triangle

This example aims at showing a minimal wgpu setup to draw a triangle.

The corner coordinates of the triangle are hardcoded in the vertex shader for simplicity.
Only the current corner index is passed to the vertex shader as input.
The steps of this minimal program are the following.

1. (async) Initialize the connection with the GPU device
2. Initialize a wgpu Texture object that will serve as a write target for our pipeline
3. Initialize a wgpu Buffer where the Texture output will be transferred to
4. Load the shader module, containing both the vertex and fragment shaders
5. Define our render pipeline, including:
   - the vertex shader
   - the fragment shade
   - the primitive type (triangle list)
6. Define our command encoder:
   1. Start by defining our render pass:
      - Link to the texture output
      - Link to the pipeline
      - Draw the primitive (provide vertices indices)
   2. Add a command to copy the texture output to the output buffer
7. Submit our commands to the device queue
8. (async) Transfer the output buffer into an image we can save to disk

## 2. Triange provided with vertex buffer

This example aims at showing vertex buffer usage in a minimal wgpu setup.
The corner coordinates of the triangle are provided via a vertex buffer
and the vertex indices in the face are provided as an index buffer.

## 3. Loading an OBJ

This example aims at showing OBJ model display in a minimal wgpu setup.
The vertices and faces (indices) are loaded from an OBJ file (bunny).
The OBJ is projected with a simple orthogonal projection in the vertex shader with some scaling.
The appearance is set to a simple white in the fragment shader.

## 4. Using the depth

This example aims at showing how to use and retrieve the depth (Z) in a minimal wgpu setup.
We reuse the bunny OBJ from the previous example,
except this time we try to output a depth map instead of just a mask of the bunny.
This example also shows the effect of the clipping space (0.0-1.0 for Z).
Indeed, a small part of the bunny ear is cut, due to negative Z coordinates.

## 5. Camera projection with uniform buffers

This example shows how to do camera projections with uniform buffers in a minimal wgpu setup.
We reuse the bunny OBJ from the previous example,
except this time we provide camera parameters to the vertex shader with uniform buffers
to perform a perspective projection instead of a simple orthogonal projection.
In addition, we also introduce the notion of near/far planes to control Z clipping.