// Compute shader that computes Conway's game of life rules for each thread/cell
//
// Note 1: we can't use 2 storage textures, one with read access, the other with write access,
// because currently the Rust implementation of webgpu does not support read-only nor read-write
// storage textures of any format (r32uint should be supported as per the WebGPU spec).
// Instead we use a regular texture as input.
// Note 2: 32 bits uint is very wastefull for storing a cell's binary liveliness, but storage
// textures are limited in their format. In a more performance minded setting we could pack
// 32 cells in a single texel.
// Note 3: there are 2 functionnaly equivalent entry points in this file, adapt the pipeline
// creation to select the one you want to use

@group(0) @binding(0) var input_grid: texture_2d<u32>;
@group(0) @binding(1) var output_grid: texture_storage_2d<r32uint, write>;

// This is a basic variant that reads directly into the input texture
// Because each threads access its 3x3 neighbnorhood, this implies redundant accesses to
// (comparatively slow) VRAM
@compute
@workgroup_size(16, 16)
fn step(@builtin(global_invocation_id) cell_id: vec3u) {

    var num_alive = 0;
    var alive: bool = false;
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let cell = textureLoad(input_grid, (vec2i(cell_id.xy) + vec2i(j, i)) % vec2i(textureDimensions(input_grid)), 0).x;
            if (i == 0) && (j == 0) {
                if cell > 0 {
                    alive = true;
                }
            } else {
                if cell > 0 {
                   num_alive++;
                }
            }
        }
    }

    if alive {
        if (num_alive < 2) || (num_alive > 3) {
            alive = false;
        } else {
            alive = true;
        } 
    } else if num_alive == 3 {
        alive = true;
    }

    textureStore(output_grid, cell_id.xy, vec4u(select(0, 1, alive)));
}

// This variant uses local workgroup memory which is faster thatn VRAM (closer in speed to an L1 cache).
// The threads in the workgroup cooperatively load the cells necessary for their computations into
// an array declared in the workgroup address space.
// This variable is common to all threads in the workgroup. Here we just access it for reading,
// otherwise concurrent accesses should be taken care of (e.g. using atomic operations).

// array<bool> would be better, but it does not seem to work
var<workgroup> grid: array<u32, 324>; // (1+16+1)x(1+16+1) = 18x18

@compute
@workgroup_size(16, 16)
fn step_local_mem(@builtin(global_invocation_id) cell_id: vec3u,
        @builtin(local_invocation_id) local_id: vec3u,
) {
    let width = textureDimensions(input_grid).x;
    let height = textureDimensions(input_grid).y;
    // Every thread loads its cell
    grid[18 * (local_id.y + 1) + (local_id.x + 1)] = textureLoad(input_grid, vec2i(cell_id.xy), 0).x;

    // Top threads also load an extra line above
    if (local_id.y == 0) {
        grid[local_id.x + 1] = textureLoad(input_grid, vec2i(i32(cell_id.x), i32((cell_id.y - 1) % height)), 0).x;
        if (local_id.x == 0) {
            grid[0] = textureLoad(input_grid, vec2i(i32((cell_id.x - 1) % width), i32((cell_id.y - 1) % height)), 0).x;
        } else if (local_id.x == 15) {
            grid[17] = textureLoad(input_grid, vec2i(i32((cell_id.x + 1) % width), i32((cell_id.y - 1) % height)), 0).x;
        }
    }
    // Bottom threads load an extra line below
    else if (local_id.y == 15) {
        grid[18 * 17 + local_id.x + 1] = textureLoad(input_grid, vec2i(i32(cell_id.x), i32((cell_id.y + 1) % height)), 0).x;
        if (local_id.x == 0) {
            grid[18 * 17] = textureLoad(input_grid, vec2i(i32((cell_id.x - 1) % width), i32((cell_id.y + 1) % height)), 0).x;
        } else if (local_id.x == 15) {
            grid[18 * 17 + 17] = textureLoad(input_grid, vec2i(i32((cell_id.x + 1) % width), i32((cell_id.y + 1) % height)), 0).x;
        }
    }
    // Left side threads load an extra cell to the left
    else if (local_id.x == 0) {
        grid[18 * local_id.y] = textureLoad(input_grid, vec2i(i32((cell_id.x - 1) % width), i32(cell_id.y)), 0).x;
    }
    // Right side threads load an extra cell to the right
    else if (local_id.x == 15) {
        grid[18 * local_id.y + 17] = textureLoad(input_grid, vec2i(i32((cell_id.x + 1) % width), i32(cell_id.y)), 0).x;
    }

    // Wait for each thread to have loaded its data
    workgroupBarrier();

    // The rest is similar to the other variant
    let cell_coord = vec2i(local_id.xy + vec2u(1, 1));
    var num_alive = 0;
    var alive: bool = false;
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let cell = grid[18 * (cell_coord.y + i) + (cell_coord.x + j)];
            if (i == 0) && (j == 0) {
                if cell > 0 {
                    alive = true;
                }
            } else {
                if cell > 0 {
                   num_alive++;
                }
            }
        }
    }

    if alive {
        if (num_alive < 2) || (num_alive > 3) {
            alive = false;
        } else {
            alive = true;
        } 
    } else if num_alive == 3 {
        alive = true;
    }

    textureStore(output_grid, cell_id.xy, vec4u(select(0, 1, alive)));
}


