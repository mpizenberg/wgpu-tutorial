// Vertex shader
// Just a very simple orthogonal projection

@vertex
fn vertex_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(7.0 * position.x, 7.0 * (position.y - 0.1), 0.0, 1.0);
}

// Fragment shader
// Just return white for model faces

@fragment
fn fragment_main(@builtin(position) clip_position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
