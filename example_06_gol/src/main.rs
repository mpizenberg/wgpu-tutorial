//! This example shows how to use a compute shader to evaluate a simple function in parallel
//! on a set of data.
//!
//! The function is the ruleset of Conway's Game of Life. It is defined on a 3x3 neighborhood
//! of pixels (or cells) from a larger grid. Its output is the new value of the center pixel/cell.
//! We use a compute shader to apply the function at all pixel locations of an input image,
//! producing a new image representing the next state of the cellular automaton.
//! We iterate the process a few times to capture the automaton evolution, each time swapping
//! the input and output images.
//!
//! This sample demonstrates:
//! * The use of compute shader that writes into a storage texture
//! * The classic "ping-pong" technique between two textures for iterative computation
//! * The use of local workgroup memory (optional: see the shader code for details)
//!
//! 1. (async) Initialize the connection with the GPU device
//! 2. Initialize 2 wgpu Texture objects that will serve as input and output data to a compute shader
//!    To be able to write to these textures from a computer shader, their declaration includes
//!    "storage binding" as usage (as well as "texture binding", for reading them)
//! 3. Place cells at random locations into the first input grid (initial automaton state)
//!    and upload this data int the respective texture.
//! 4. Initialize a wgpu Buffer where one of the texture will be transferred to
//! 5. Load the shader module, containing a compute shader
//! 6. Define a compute pipeline
//! 7. Create two similar bind groups:
//!    - one that binds texture 0 as input and texture 1 as output
//!    - the other, similar but with bindings swapped
//! 8. Iterate multiple run of our compute shader
//!    1. dispatch a grid of workgroups running the computer kernel to cover the whole grid
//!    2. copy the current input grid into the staging buffer
//!    3. submit the commands to the queue
//!    4. read back the content of the staging buffer
//!    5. postprocess the pixel data to encode a frame in an animated gif
//!    6. unmap the staging buffer so that it's reusable for next iteration

use rand::prelude::*;
use std::borrow::Cow;
use std::fs::File;

fn main() {
    // Make the main async
    pollster::block_on(run());
}

async fn run() {
    // (1) Initializing WebGPU
    println!("Initializing WebGPU ...");
    let (device, queue) = init_wgpu_device().await.unwrap();

    // (2) Initialize two textures for the input (current state) and output (next state) grids
    let width = 256;
    let height = 256;
    let grids = [
        init_grid_texture(&device, width, height),
        init_grid_texture(&device, width, height),
    ];
    let desc = wgpu::TextureViewDescriptor {
        label: None,
        format: None,
        dimension: None,
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(1),
    };
    let grid_views = [grids[0].create_view(&desc), grids[1].create_view(&desc)];

    // (3) Put some initial random cells into the input grid
    let mut rng = rand::thread_rng();
    let init_state: Vec<u8> = (0..4 * width * height)
        .map(|i| {
            if rng.gen::<u8>() > 200 && (i % 4 == 0) {
                1
            } else {
                0
            }
        })
        .collect();
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &grids[0],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &init_state,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * width), // texel size of r32uint texture is 4 bytes
            rows_per_image: Some(height),
        },
        grids[0].size(),
    );

    // (4) Create a staging buffer for retrieving the content of the current state texture
    let staging_buffer = init_buffer(&device, width, height);

    // (5) Initialize the shader module, containing a single computer shader
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Game of Life"),
        source: wgpu::ShaderSource::Wgsl(include_str!("game_of_life.wgsl").into()),
    });

    // (6) Define a compute pipeline
    // Use "step" or "step_local_mem" as the entry point of the shader.
    let pipeline = build_pipeline(&device, &shader_module, "step");

    // (7) create two bind groups alternating the role of the 2 textures
    let desc = wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&grid_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&grid_views[1]),
            },
        ],
    };
    let desc_alt = wgpu::BindGroupDescriptor {
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&grid_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&grid_views[0]),
            },
        ],
        ..desc
    };
    let bind_groups = [
        device.create_bind_group(&desc),
        device.create_bind_group(&desc_alt),
    ];

    // Show progress in the console
    const N_ITERS: usize = 1000;
    let bar = indicatif::ProgressBar::new(N_ITERS as u64);

    // Animated gif encoder
    let mut image = File::create("image.gif").unwrap();
    let color_map = &[0x11, 0x77, 0xaa, /* cell color */
                      0x33, 0x22, 0,    /* background color*/ ];
    let mut gif_enc =
        gif::Encoder::new(&mut image, width as u16, height as u16, color_map).unwrap();

    // (8) Evolve the automaton over some iterations
    println!("Computing Game of Life's iterations ...");
    for i in 0..N_ITERS {
        bar.inc(1);

        // Initialize a command encoder
        let mut encoder = device.create_command_encoder(&Default::default());

        // (8.1) launch the compute kernel in blocks
        launch_kernel(
            &mut encoder,
            &pipeline,
            &bind_groups[i % 2], // alternate input and output grids every other iteration
            &grids[0].size(),
        );

        // (8.2) Copy the current input texture to the staging buffer
        copy_texture_to_buffer(&mut encoder, &grids[i % 2], &staging_buffer);

        // (8.3) Finalize the command encoder and send it to the queue
        queue.submit(Some(encoder.finish()));

        // (8.4) read back the content of the staging buffer
        let data = readback_buffer_data(&device, &staging_buffer).await;

        // (8.5) Convert the returned data and encode it as frame of the result gif
        // u32 texel -> u8 \in {0, 1} (gif is a paletted format)
        let pixels: Vec<u8> = bytemuck::cast_slice::<u8, u32>(&data as &[u8])
            .iter()
            .map(|&x| if x > 0 { 0u8 } else { 1u8 })
            .collect();

        let mut frame = gif::Frame::default();
        frame.delay = 2;
        frame.width = width as u16;
        frame.height = height as u16;
        frame.buffer = Cow::Borrowed(&*pixels);
        gif_enc.write_frame(&frame).unwrap();

        // (8.6) unmap the buffer to allow subsequent GPU writes to it
        drop(data); // CPU pointer must not survive unmap (Rust memory safety guarantee)
        staging_buffer.unmap();
    }
    bar.finish();

    println!("Terminating the program ...")
}

/// (1) Initializing WebGPU
async fn init_wgpu_device() -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
    // Start an "Instance", which is the context for all things wgpu.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    // An "Adapter" is a handle to a physical graphics/compute device.
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(), // None | LowPower | HighPerformance
            compatible_surface: None,
            force_fallback_adapter: false, // If needed to force CPU fallback?
        })
        .await
        .unwrap();

    // Request a connection to a physical device,
    // and also access to the queue for its command buffers.
    // It is possible, if necessary, to add a description of required features.
    adapter.request_device(&Default::default(), None).await
}

// (2) Initialize a texture for the grid of cells.
// Using u32 for storing boolean cell liveliness is a waste, but smaller format (e.g. u8) are
// not allowed currently for storage textures.
fn init_grid_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    let desc = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[wgpu::TextureFormat::R32Uint],
    };

    device.create_texture(&desc)
}

/// (4) Initialize a staging buffer. Required because we can't retrieve the
/// texture content directly when they're used as pipeline data.
fn init_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
    let desc = wgpu::BufferDescriptor {
        label: None,
        size: (width * height * 4).into(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    };

    device.create_buffer(&desc)
}

/// (6) Initialize a compute pipeline. Note how it requires a lot less settings compared
/// to a render pipeline (less fixed functionalities to configure)
fn build_pipeline(
    device: &wgpu::Device,
    shader_module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let desc = wgpu::ComputePipelineDescriptor {
        label: Some("Game of Life"),
        layout: None,
        module: shader_module,
        entry_point,
    };

    device.create_compute_pipeline(&desc)
}

/// (8.1) Dispatch our compute shader into thread groups (aka workgroups), as many as required to cover the whole
/// grid (thread per cell/texel)
fn launch_kernel(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    grids_bind_group: &wgpu::BindGroup,
    grid_size: &wgpu::Extent3d,
) {
    // Setup a compute pass
    let desc = wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    };
    let mut pass = encoder.begin_compute_pass(&desc);

    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, grids_bind_group, &[]);

    // Workgroups are arranged in a 16x16 grid (as declared in the shader code),
    // so we need to launch (width/16)x(height/16) of them
    pass.dispatch_workgroups(grid_size.width / 16, grid_size.height / 16, 1);
}

/// (8.2) Copy the texture output into a buffer
fn copy_texture_to_buffer(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    output_buffer: &wgpu::Buffer,
) {
    let texel_size = texture.format().block_copy_size(None).unwrap();
    encoder.copy_texture_to_buffer(
        // source
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        // destination
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(texel_size * texture.width()),
                rows_per_image: Some(texture.height()),
            },
        },
        // copy_size
        texture.size(),
    );
}

/// (8.4) Retrieve the buffer data from the GPU
async fn readback_buffer_data<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
) -> wgpu::BufferView<'a> {
    let buffer_slice = buffer.slice(..);

    // request the buffer to be mapped (made visible) to CPU memory
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

    // block on the device until the mapping callback above is called (indicating completion)
    device.poll(wgpu::Maintain::Wait);

    // it's now safe to retrieve a pointer to the mapped memory
    buffer_slice.get_mapped_range()
}
