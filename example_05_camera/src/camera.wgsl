// Vertex shader
//
// We compute a perspective camera projection.
// Remark that this should be done once on the CPU
// instead of for each vertex in the vertex shader.
// But for this minimal example it's fine.

/// Perspective projection parameters
struct ProjCamParams {
    focal: f32,
    ratio: f32,
    near: f32,
    far: f32,
}

@group(0) @binding(0) var<uniform> proj_cam_params: ProjCamParams;

@vertex
fn vertex_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    let focal = proj_cam_params.focal;
    let ratio = proj_cam_params.ratio;
    let near = proj_cam_params.near;
    let far = proj_cam_params.far;

    // Build the camera view and projection matrices.
    //
    // In GPU programs, the extrinsics parameters are usually called the view matrix,
    // and the intrinsics are embedded in the projection matrix, which also deals with near/far clipping.
    // In addition, going from local coordinates to world coordinates,
    // we often have a model matrix applied to each vertex local coordinates.
    // So in the end, the projection pipeline looks as follows:
    //
    // clip_coords = proj_mat * view_mat * model_mat * homogeneous(vertex_coords)
    //
    // The LearnWebGPU website provides an [excellent explanation](https://eliemichel.github.io/LearnWebGPU/basic-3d-rendering/3d-meshes/projection-matrices.html)
    // for how to build the projection matrix, both for orthographic or perspective projections.
    // The summary is that, given a focal length, aspect ratio, near and far clipping planes,
    // the perspective projection matrix has the following shape:
    // 
    // ```txt
    // focal, 0.0          , 0.0               , 0.0                       ,
    // 0.0  , focal * ratio, 0.0               , 0.0                       ,
    // 0.0  , 0.0          , far / (far - near), -far * near / (far - near),
    // 0.0  , 0.0          , 1.0               , 0.0                       ,
    // ```
    //
    // The near and far clipping planes enable the GPU to limit to a restricted depth of interest
    // the amount of vertices that also need to be processed in the fragment shader .
    // If the near/far range is too wide, it may also degrade the floating point precision.
    //
    // > WARNING: note that `near` can be very small but not exactly 0.0.
    // > Otherwise, the homogeneous component W turns into 0.
    // 
    // > Note: WGSL uses column-major matrices,
    // > so yo might need to transpose your matrices if you build them in row-major.
    let proj_mat = transpose(mat4x4f(
        focal, 0.0          , 0.0               , 0.0                       ,
        0.0  , focal * ratio, 0.0               , 0.0                       ,
        0.0  , 0.0          , far / (far - near), -far * near / (far - near),
        0.0  , 0.0          , 1.0               , 0.0                       ,
    ));

    // Build the view matrix (extrinsics)
    let view_mat = transpose(mat4x4f(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ));

    // Build the model matrix applying a small translation.
    let model_mat = transpose(mat4x4f(
        1.0, 0.0, 0.0,  0.0,
        0.0, 1.0, 0.0, -0.1, // y - 0.1
        0.0, 0.0, 1.0,  0.5, // z + 0.5
        0.0, 0.0, 0.0,  1.0,
    ));

    let homogeneous_pos = vec4<f32>(position, 1.0);
    return proj_mat * view_mat * model_mat * homogeneous_pos;
}

// Fragment shader
// Output the normalized Z clip coordinate

@fragment
fn fragment_main(@builtin(position) clip_pos: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(vec3<f32>(clip_pos.z), 1.0);
}
