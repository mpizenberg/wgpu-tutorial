//! This example aims at showing vertex buffer usage in a minimal wgpu setup.
//! 
//! The corner coordinates of the triangle are provided via a vertex buffer
//! and the vertex indices in the face are provided as an index buffer.
//! The steps of this minimal program are the following.
//!
//! 1. (async) Initialize the connection with the GPU device
//! 2. Initialize a wgpu Texture object that will serve as a write target for our pipeline
//! 3. Initialize a wgpu Buffer where the Texture output will be transferred to
//! 4. **(new)** Create and initialize a vertex buffer containing the triangle coordinates
//! 5. **(new)** Create and initialize an index buffer containing the vertex indices in the face
//! 6. Load the shader module, containing both the vertex and fragment shaders
//! 7. Define our render pipeline, including:
//!    - the vertex shader: **(new)** include our vertex buffer layout
//!    - the fragment shader
//!    - the primitive type (triangle list)
//! 8. Define our command encoder:
//!    1. Start by defining our render pass:
//!       - Link to the texture output
//!       - Link to the pipeline
//!       - **(new)** Provide vertex buffer and index buffer
//!       - Draw the primitive
//!    2. Add a command to copy the texture output to the output buffer
//! 9. Submit our commands to the device queue
//! 10. (async) Transfer the output buffer into an image we can save to disk

use wgpu::util::DeviceExt; // Utility trait to create and initialize buffers with device.create_buffer_init()

fn main() {
    // Make the main async
    pollster::block_on(run());
}

async fn run() {
    // (1) Initializing WebGPU
    println!("Initializing WebGPU ...");
    let (device, queue) = init_wgpu_device().await.unwrap();

    // (2) Initialize the output texture
    let texture = init_output_texture(&device, 256);
    let texture_view = texture.create_view(&Default::default());

    // (3) Initialize a buffer for the texture output
    let output_buffer_desc = create_texture_buffer_descriptor(&texture);
    let output_buffer = device.create_buffer(&output_buffer_desc);

    // (4) Create and initialize the vertex buffer for the triangle vertices
    // (needs the DeviceExt trait)
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // (5) Create and initialize the index buffer for the indices of the triangle face
    // (needs the DeviceExt trait)
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(INDICES),
        usage: wgpu::BufferUsages::INDEX,
    });

    // (6) Load the shader module, containing both the vertex and fragment shaders
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("triangle_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("triangle.wgsl").into()),
    });

    // (7) Define our pipeline
    let pipeline = build_simple_pipeline(
        &device,
        &shader_module,
        texture.format(),
        Vertex::buffer_layout(),
    );

    // Initialize a command encoder
    let mut encoder = device.create_command_encoder(&Default::default());

    // (8.1) Draw our pipeline (add render pass to the command encoder)
    // This needs to be inside {...} or a function so that the &pipeline lifetime works.
    draw_pipeline(
        &mut encoder,
        &pipeline,
        &texture_view,
        &vertex_buffer,
        &index_buffer,
        INDICES.len() as u32,
    );

    // (8.2) Copy the texture output into a buffer
    copy_texture_to_buffer(&mut encoder, &texture, &output_buffer);

    // (9) Finalize the command encoder and send it to the queue
    println!("Submitting commands to the queue ...");
    queue.submit(Some(encoder.finish()));

    // (10) Transfer the texture output buffer into an image buffer
    println!("Saving the GPU output into an image ...");
    let img = to_image(&device, &output_buffer, texture.width(), texture.height()).await;

    println!("Saving the image to disk ...");
    img.save("image.png").unwrap();

    // Flushes any pending write operations and unmaps the buffer from host memory
    output_buffer.unmap();

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

/// (2) Initialize the output texture
fn init_output_texture(device: &wgpu::Device, texture_size: u32) -> wgpu::Texture {
    let texture_desc = wgpu::TextureDescriptor {
        label: Some("output_texture"),
        // The texture size. (layers is set to 1)
        size: wgpu::Extent3d {
            width: texture_size,
            height: texture_size,
            depth_or_array_layers: 1,
        },
        dimension: wgpu::TextureDimension::D2,
        mip_level_count: 1, // the number of mip levels the texture will contain
        sample_count: 1,    // sample_count > 1 would indicate a multisampled texture
        // Use RGBA format for the output
        format: wgpu::TextureFormat::Rgba8Unorm,
        // RENDER_ATTACHMENT -> so that the GPU can render to the texture
        // COPY_SRC -> so that we can pull data out of the texture
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        // Specify the allowed formats when calling "texture.create_view()"
        view_formats: &[],
    };
    device.create_texture(&texture_desc)
}

/// (3) Create a buffer descriptor of the correct size for the texture
fn create_texture_buffer_descriptor(texture: &wgpu::Texture) -> wgpu::BufferDescriptor {
    let texel_size = texture.format().block_copy_size(None).unwrap();
    wgpu::BufferDescriptor {
        size: (texel_size * texture.width() * texture.height()).into(),
        usage: wgpu::BufferUsages::COPY_DST
            // this tells wpgu that we want to read this buffer from the cpu
            | wgpu::BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    }
}

/// Each corner of the triangle is a Vertex with the following properties
/// Bytemuck is used to enable easy casting to a &[u8].
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

// Indices and vertices of the triangle
const INDICES: &[u16] = &[0, 1, 2];
const VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

impl Vertex {
    /// Define the layout of Vertex buffers
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            // array_stride is the bytes count between two vertices
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// (7) Define our simple render pipeline
fn build_simple_pipeline(
    device: &wgpu::Device,
    shader_module: &wgpu::ShaderModule,
    texture_format: wgpu::TextureFormat,
    vertex_buffer_layout: wgpu::VertexBufferLayout,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: None, // "auto"
        vertex: wgpu::VertexState {
            module: shader_module,
            entry_point: "vertex_main",
            buffers: &[vertex_buffer_layout],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader_module,
            entry_point: "fragment_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        // If the pipeline will be used with a multiview render pass, this
        // indicates how many array layers the attachments will have.
        multiview: None,
    })
}

/// (8.1) Draw our pipeline (add render pass to the command encoder).
fn draw_pipeline(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    texture_view: &wgpu::TextureView,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    num_indices: u32,
) {
    // Setup the pass that will render into our texture
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &texture_view,  // the output texture
            resolve_target: None, // only useful for multisampling
            ops: wgpu::Operations {
                // "load" specifies how data is read.
                // "Clear" is the lightest way to initialize the texture.
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                }),
                // whether data is written to or not. Store | Discard
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });

    // Draw the render pass for our pipeline
    render_pass.set_pipeline(&pipeline);
    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    render_pass.draw_indexed(0..num_indices, 0, 0..1);
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

/// (10) Copy the texture output buffer into an image buffer
async fn to_image(
    device: &wgpu::Device,
    texture_buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
) -> image::RgbaImage {
    let buffer_slice = texture_buffer.slice(..);

    // NOTE: We have to create the mapping THEN device.poll() before await the future.
    // Otherwise the application will freeze.
    let (tx, rx) = oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.await.unwrap().unwrap();

    // Synchronously and immediately map a buffer for reading.
    // Will panic if buffer_slice.map_async() did not finish yet.
    let data = buffer_slice.get_mapped_range();

    image::RgbaImage::from_raw(width, height, Vec::from(&data as &[u8])).unwrap()
}
