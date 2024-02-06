// Vertex shader
// Just a very simple orthogonal projection,
// and also outputs the Z coordinate, scaled so that they roughly span the 0.0-1.0 range.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) z: f32,
};

@vertex
fn vertex_main(@location(0) position: vec3<f32>) -> VertexOutput {
    return VertexOutput(
        // clip_position
        vec4<f32>(7.0 * position.x, 7.0 * (position.y - 0.1), 14.0 * (position.z + 0.05), 1.0),
        // z
        14.0 * (position.z + 0.05),
    );
}

// Fragment shader
// Output the depth

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(vec3<f32>(in.z), 1.0);
    // return vec4<f32>(1.0);
}
