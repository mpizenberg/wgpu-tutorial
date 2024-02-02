/// Vertex shader
/// It takes as input the index of the current vertex being processed,
/// and outputs the 2D position of the corresponding triangle corner.
@vertex fn vertex_main(@builtin(vertex_index) index : u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2(0.0, 0.5),
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5)
    );
  return vec4f(corners[index], 0.0, 1.0);
}

/// Fragment shader
/// This just returns the same redish color
/// for every pixel inside the triangle.
@fragment fn fragment_main() -> @location(0) vec4f {
    return vec4f(1.0, 0.4, 0.1, 1.0);
}