use wgpu::util::DeviceExt; // Utility trait to create and initialize buffers with device.create_buffer_init()

fn main() {
    // Make the main async
    pollster::block_on(run());
}

async fn run() {
    // Initializing WebGPU
    println!("Initializing WebGPU ...");
    let (device, queue) = init_wgpu_device().await.unwrap();

    // Initialize the output texture
    let texture = init_output_texture(&device, 256);
    let texture_view = texture.create_view(&Default::default());

    // Load the OBJ bunny
    let (models, _) = tobj::load_obj("bunny.obj", &tobj::GPU_LOAD_OPTIONS).unwrap();
    let bunny = &models[0].mesh;

    // Create and initialize the vertex buffer for the vertices in the bunny mesh
    // (needs the DeviceExt trait)
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&bunny.positions),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Create and initialize the index buffer for the indices of the vertices in the bunny mesh
    // (needs the DeviceExt trait)
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&bunny.indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Define our pipeline
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("triangle_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("obj.wgsl").into()),
    });
    let pipeline = build_simple_pipeline(
        &device,
        &shader_module,
        texture.format(),
        vtx_buffer_layout(),
    );

    // Initialize a command encoder
    let mut encoder = device.create_command_encoder(&Default::default());

    // Draw our pipeline (add render pass to the command encoder)
    // This needs to be inside {...} or a function so that the &pipeline lifetime works.
    draw_pipeline(
        &mut encoder,
        &pipeline,
        &texture_view,
        &vertex_buffer,
        &index_buffer,
        bunny.indices.len() as u32,
    );

    // Initialize a buffer for the texture output
    let output_buffer_desc = create_texture_buffer_descriptor(&texture);
    let output_buffer = device.create_buffer(&output_buffer_desc);

    // Copy the texture output into a buffer
    copy_texture_to_buffer(&mut encoder, &texture, &output_buffer);

    // Finalize the command encoder and send it to the queue
    println!("Submitting commands to the queue ...");
    queue.submit(Some(encoder.finish()));

    // Transfer the texture output buffer into an image buffer
    println!("Saving the GPU output into an image ...");
    let img = to_image(&device, &output_buffer, texture.width(), texture.height()).await;

    println!("Saving the image to disk ...");
    img.save("image.png").unwrap();

    // Flushes any pending write operations and unmaps the buffer from host memory
    output_buffer.unmap();

    println!("Terminating the program ...")
}

/// Initializing WebGPU
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

/// Initialize the output texture
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

/// Define the layout of Vertex buffers
pub fn vtx_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        // array_stride is the bytes count between two vertices
        array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // position
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// Define our simple render pipeline
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
            cull_mode: Some(wgpu::Face::Front),
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

/// Draw our pipeline (add render pass to the command encoder).
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
    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    render_pass.draw_indexed(0..num_indices, 0, 0..1);
}

/// Create a buffer descriptor of the correct size for the texture
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

/// Copy the texture output into a buffer
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

/// Copy the texture output buffer into an image buffer
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
