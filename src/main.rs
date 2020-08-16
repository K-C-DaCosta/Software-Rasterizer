use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use std::collections::HashMap;
use std::f32;
use std::mem;
use std::thread;
use std::time;

use software_rasterizer::containers::*;

use nalgebra::{Matrix4, Rotation3, Translation3, Unit, Vector2, Vector3, Vector4};

use software_rasterizer::math_util::fixed_point::*;
use software_rasterizer::{
    bounding::*,
    graphics,
    graphics::*,
    image,
    image::{
        buffer::{RowBuffer2D, TiledBuffer2D},
        sampler::{LinearSampler2D, NearestSampler2D, Sampler2D},
        Buffer2D,
    },
    math_util::{Camera, MatrixStack},
    model,
    system::video::Window,
    time_util::*,
};

static mut PAUSE: bool = false;

#[allow(unused_variables)]
static FRAG_SHADER: fn(
    &(u32, u32),
    &Vector4<f32>,
    &TriangleData,
    &ShaderGlobals,
    &mut RowBuffer2D<u32>,
    &mut OutputBuffers,
) -> (u16, u32) = |pixel_pos: &(u32, u32),
                   weights: &Vector4<f32>,
                   attr: &TriangleData,
                   glbs: &ShaderGlobals,
                   color_buffer: &mut RowBuffer2D<u32>,
                   aux_buffers: &mut OutputBuffers| {
    let pixel_pos = *pixel_pos;
    let adj_pos = (pixel_pos.0, pixel_pos.1 + 1);

    //grab depth attributes
    let depths = attr.get_scalar_standard(GlobalIndex::VertexDepths);
    //grab depth_buffer
    let depth_buffer =
        unsafe { &mut *aux_buffers.get_scalar_buffer(GlobalIndex::PrimaryDepthBuffer) };

    //perspective correct depth calculation
    let current_depth = depth_buffer.get(pixel_pos);
    let new_depth = ((depths[0] * weights.x + depths[1] * weights.y + depths[2] * weights.z).abs()
        * (65535.0 / 512.0)) as u16;

    //branchless depth calculation
    let depth_mask: u32 =
        unsafe { std::mem::transmute((new_depth as i32 - current_depth as i32) >> 31) };
    let final_depth_bits = !depth_mask & current_depth as u32 | depth_mask & new_depth as u32;
    let final_depth = final_depth_bits;

    depth_buffer.set(pixel_pos, final_depth as u16);

    //get positions
    // let pos_lst = attr.get(0);
    // let position = pos_lst[0] * weights[0] + pos_lst[1] * weights[1] + pos_lst[2] * weights[2];

    //get normals
    // let normals = attr.get(1);
    // let normal = normals[0] * weights[0] + normals[1] * weights[1] + normals[2] * weights[2];

    // if depth_mask > 0{
    //     // linearly interpolate uvs
    //     let uvs = attr.get_standard(GlobalIndex::UV);
    //     let uv = uvs[0] * weights[0] + uvs[1] * weights[1] + uvs[2] * weights[2];

    //     //grab texture
    //     let texture = unsafe { &mut *aux_buffers.texture_buffers[0].unwrap() };
    //     let new_color = <NearestSampler2D as Sampler2D<f32, u32>>::uv_map(&NearestSampler2D, uv, texture);
    //     color_buffer.set(pixel_pos, new_color);
    // }

    //branchless sampling
    let uvs = attr.get_standard(GlobalIndex::UV);
    let uv = uvs[0] * weights[0] + uvs[1] * weights[1] + uvs[2] * weights[2];
    //grab texture
    let texture = unsafe { &mut *aux_buffers.texture_buffers[0].unwrap() };

    let current_color = color_buffer.get(pixel_pos);
    let new_color =
        <NearestSampler2D as Sampler2D<f32, u32>>::uv_map(&NearestSampler2D, uv, texture);

    let final_color = !depth_mask & current_color | depth_mask & new_color;
    color_buffer.set(pixel_pos, final_color);
    (final_depth as u16, final_color)

    // color_buffer.set(pixel_pos,new_color);
    // (0,new_color)
};

#[allow(unused_variables)]
static VERT_SHADER: fn(&mut TriangleData, &ShaderGlobals, &MatrixStack) =
    |attribs: &mut TriangleData, globals: &ShaderGlobals, stack: &MatrixStack| {
        let mut vert;
        for v in attribs.get_mut(0) {
            vert = *v;
            vert = stack.modelview * vert;
            *v = vert;
        }

        for n in attribs.get_mut(1) {
            vert = *n;
            vert = stack.normalmat * vert;
            *n = vert;
        }
    };

/**
 * How rasterization should work below:
 * --------------------------------------------------
 * let window = Window::new("my rasterizer",1920,1080...);                    [x]
 * let model = Model::load_model("cube.obj");                                 [ ]
 * let matrix_stack = MatrixStack::new();                                     [x]
 * model.render(&matrix_stack,window.color_buffer(),window.depth_buffer());   [x]
 */
fn main() {
    let mut window = Window::new(1024, 1024, "software rasterizer: by khadeem dacosta".to_string());
    let mut sw = StopWatch::new();
    let mut timer = Timer::new();

    #[allow(unused_variables)]
    let (mut x_pos, mut y_pos) = (500.0f32, 500.0f32);

    let mut stack = MatrixStack::new();
    let mut globals = ShaderGlobals::new();
    let mut aux_buffers = OutputBuffers::new();
    let model = model::StaticModel::load_model("./assets/cube.obj").unwrap();

    let mut depth_buffer = RowBuffer2D::<u16>::new();
    depth_buffer.allocate(window.width(), window.height(), 0);

    let mut texture = image::load_texture("./assets/crate.png").unwrap();
    let mut camera = Camera::new();
    let mut key_map: HashMap<i32, bool> = HashMap::new();

    //init stack
    stack.update_viewport(window.width(), window.height());

    // window.context.mouse().set_relative_mouse_mode(true);

    //assign textures
    aux_buffers.scalar_buffers[0] = Some(&mut depth_buffer as *mut RowBuffer2D<u16>);
    aux_buffers.texture_buffers[0] = Some((&mut texture) as *mut RowBuffer2D<u32>);

    'running: loop {
        let _mouse_state = sdl2::mouse::MouseState::new(window.get_event_pump().as_mut());

        if process_events(&mut window, &mut stack, &mut camera, &mut key_map) {
            break 'running;
        }

        camera.generic_keyboard_handling(&key_map, &mut stack);

        //draw calls begin here
        sw.start();

        // //clear buffers
        window.buffer().fill_zero();
        depth_buffer.set_all(!0);

        stack.push_translate(Vector3::new(0., 0., -30.));
        stack.update();
        model.render(
            &stack,
            &globals,
            window.buffer(),
            &mut aux_buffers,
            VERT_SHADER,
            FRAG_SHADER,
        );
        stack.pop();

        stack.push_rotate(Vector3::z(), globals.scalar_table[0]);
        stack.push_translate(Vector3::new(3., 0., -30.));
        stack.update();
        model.render(
            &stack,
            &globals,
            window.buffer(),
            &mut aux_buffers,
            VERT_SHADER,
            FRAG_SHADER,
        );
        stack.pop();
        stack.pop();

        if unsafe { PAUSE } == false {
            globals.scalar_table[0] += 0.01;
        }
        globals.scalar_table[1] = 0.; //x_pos/(window.width() as f32);

        //apply changes
        window.update();

        sw.stop();

        
        timer.execute(1000, || {
            println!("metrics:{}", sw);
            sw.reset();
        });

        x_pos = _mouse_state.x() as f32;
        y_pos = _mouse_state.y() as f32;
    }
}

fn process_events(
    window: &mut Window,
    stack: &mut MatrixStack,
    camera: &mut Camera,
    key_map: &mut HashMap<i32, bool>,
) -> bool {
    for event in window.get_event_pump().as_mut().poll_iter() {
        match event {
            Event::Window { win_event: e, .. } => match e {
                WindowEvent::Resized(w, h) => {
                    window.resize(w, h);
                    stack.update_viewport(w as u32, h as u32);
                }
                _ => (),
            },
            Event::MouseMotion {
                xrel: dx, yrel: dy, ..
            } => {
                let motion_x = dx.abs() != 0;
                let motion_y = dy.abs() != 0;
                let dx = dx as f32 * -0.005;
                let dy = dy as f32 * -0.005;

                if motion_x {
                    let axis = Unit::new_unchecked(camera.q.xyz());
                    let rot = Rotation3::from_axis_angle(&axis, dx).to_homogeneous();
                    camera.r = rot * camera.r;
                    camera.update(&Vector4::y());
                    stack.view = camera.compute_view_mat();
                }

                if motion_y {
                    let axis = Unit::new_unchecked(camera.p.xyz());
                    let rot = Rotation3::from_axis_angle(&axis, dy).to_homogeneous();
                    camera.r = rot * camera.r;
                    camera.update(&Vector4::y());
                    stack.view = camera.compute_view_mat();
                }
            }
            Event::Quit { .. }
            | Event::KeyDown {
                keycode: Some(Keycode::Escape),
                ..
            } => return true,

            Event::KeyDown {
                keycode: Some(key_int),
                ..
            } => {
                match key_int {
                    Keycode::Space => unsafe {
                        //graphics::TRIANGLE_MASK = !graphics::TRIANGLE_MASK;

                        unsafe {
                            graphics::COUNTER += 1;
                        }
                    },
                    Keycode::P => unsafe {
                        PAUSE = !PAUSE;
                    },
                    _ => (),
                }

                key_map.insert(key_int as i32, true);
            }
            Event::KeyUp {
                keycode: Some(key_int),
                ..
            } => {
                key_map.insert(key_int as i32, false);
            }
            _ => {}
        }
    }
    false
}
