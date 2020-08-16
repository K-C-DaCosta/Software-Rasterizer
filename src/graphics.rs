use std::arch::x86_64::*;
use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::mem::*;
use std::ops::{Index, IndexMut};

use super::{
    bounding::*,
    image::{
        buffer::{RowBuffer2D, TiledBuffer2D},
        sampler,
        sampler::{LinearSampler2D, NearestSampler2D, Sampler2D},
        Buffer2D,
    },
    math_util::{fixed_point::*, geometry::*},
};

use nalgebra::{Matrix4, Vector2, Vector4};

pub static mut TRIANGLE_MASK: bool = false;
pub static mut COUNTER: i64 = 0;
const MAX_ATTRIBUTES: usize = 12;
const MAX_USED: usize = MAX_ATTRIBUTES / 3;
const MAX_GLOBALS: usize = 4;

///Standard index values for certain common attributes
pub enum GlobalIndex {
    ViewPort,
    Projection,
    Eye,
    Model,

    //triangle vertex attribues
    Position,
    Normal,
    UV,
    VertexDepths,

    PrimaryColorBuffer,
    PrimaryDepthBuffer,
}

impl GlobalIndex {
    pub fn to_usize(&self) -> usize {
        match self {
            //GlobalVariable standard indicies
            Self::ViewPort => 0,
            Self::Projection => 1,
            Self::Eye => 2,
            Self::Model => 3,

            //vertex attribute indicies
            Self::Position => 0,
            Self::Normal => 1,
            Self::UV => 2,
            //vertexdepths is zero because it should exits in the
            //scalar table in the vertattrib struct
            Self::VertexDepths => 0,

            //OutputBuffers indicies
            Self::PrimaryColorBuffer => 0,
            Self::PrimaryDepthBuffer => 0,
        }
    }
}

type ColorBufferType = RowBuffer2D<u32>;
type ScalarBufferType = RowBuffer2D<u16>;
type TextureBufferType = RowBuffer2D<u32>;

pub struct OutputBuffers {
    pub color_buffers: [Option<*mut ColorBufferType>; 2],
    pub scalar_buffers: [Option<*mut ScalarBufferType>; 2],
    pub texture_buffers: [Option<*mut TextureBufferType>; 2],
}

pub struct BufferPtrMut<T> {
    ptr: usize,
    _marker: PhantomData<T>,
}

impl<T> BufferPtrMut<T> {
    pub fn new(ptr: *mut T) -> BufferPtrMut<T> {
        unsafe {
            BufferPtrMut {
                ptr: mem::transmute(ptr),
                _marker: PhantomData::default(),
            }
        }
    }
    pub fn deref(&self) -> &mut T {
        unsafe { mem::transmute(self.ptr) }
    }
}

impl OutputBuffers {
    pub fn new() -> OutputBuffers {
        OutputBuffers {
            color_buffers: [None, None],
            scalar_buffers: [None, None],
            texture_buffers: [None, None],
        }
    }

    pub fn get_color_buffer(&self, index: GlobalIndex) -> *mut ColorBufferType {
        self.color_buffers[index.to_usize()].unwrap()
    }

    pub fn get_scalar_buffer(&self, index: GlobalIndex) -> *mut ScalarBufferType {
        self.scalar_buffers[index.to_usize()].unwrap()
    }

    pub fn get_texture_buffer(&self, index: GlobalIndex) -> *mut TextureBufferType {
        self.texture_buffers[index.to_usize()].unwrap()
    }
}

pub struct ShaderGlobals<'a, 'b> {
    pub mat_table: [Matrix4<f32>; MAX_GLOBALS],
    pub scalar_table: [f32; MAX_GLOBALS],
    pub row_texture_table: [Option<&'a RowBuffer2D<u32>>; MAX_GLOBALS],
    pub tile_texture_table: [Option<&'b TiledBuffer2D<u32>>; MAX_GLOBALS],
}

impl<'a, 'b> ShaderGlobals<'a, 'b> {
    pub fn new() -> ShaderGlobals<'a, 'b> {
        ShaderGlobals {
            mat_table: [Matrix4::identity(); MAX_GLOBALS],
            scalar_table: [0.0; MAX_GLOBALS],
            row_texture_table: [None, None, None, None],
            tile_texture_table: [None, None, None, None],
        }
    }
}

impl<'a, 'b> Index<GlobalIndex> for ShaderGlobals<'a, 'b> {
    type Output = Matrix4<f32>;
    fn index(&self, index_enum: GlobalIndex) -> &Self::Output {
        &self.mat_table[index_enum.to_usize()]
    }
}

#[derive(Copy, Clone)]
pub union AttribData {
    pub v4_float: Vector4<f32>,
    pub v4_fixed_vert: Vector4<fpi24_7>,
    pub v4_fixed_norm: Vector4<fpi3_28>,
    pub v4_fixed_uv: Vector4<fpi3_28>,
}

#[derive(Copy, Clone)]
pub struct TriangleData {
    pub data: [AttribData; MAX_ATTRIBUTES],
    pub scalar_data: [f32; MAX_ATTRIBUTES],
}

impl Default for AttribData {
    fn default() -> Self {
        AttribData {
            v4_float: Vector4::zeros(),
        }
    }
}

impl TriangleData {
    pub fn new() -> TriangleData {
        TriangleData {
            data: [AttribData::default(); MAX_ATTRIBUTES],
            scalar_data: [0.0; MAX_ATTRIBUTES],
        }
    }

    pub fn get(&self, index: usize) -> &[Vector4<f32>; 3] {
        let s_index = index * 3; //index.min(MAX_USED - 1) * 3;
        unsafe { transmute(&self.data[s_index]) }
    }

    pub fn get_standard(&self, index: GlobalIndex) -> &[Vector4<f32>; 3] {
        self.get(index.to_usize())
    }

    pub fn get_mut(&mut self, index: usize) -> &mut [Vector4<f32>; 3] {
        let s_index = index.min(MAX_USED - 1) * 3;
        unsafe { transmute(&mut self.data[s_index]) }
    }

    pub fn get_standard_mut(&mut self, index: GlobalIndex) -> &mut [Vector4<f32>; 3] {
        self.get_mut(index.to_usize())
    }

    pub fn get_scalar(&self, index: usize) -> &[f32; 3] {
        let s_index = index.min(MAX_USED - 1) * 3;
        unsafe { transmute(&self.scalar_data[s_index]) }
    }

    pub fn get_scalar_standard(&self, index: GlobalIndex) -> &[f32; 3] {
        self.get_scalar(index.to_usize())
    }

    pub fn get_scalar_mut(&mut self, index: usize) -> &mut [f32; 3] {
        let s_index = index.min(MAX_USED - 1) * 3;
        unsafe { transmute(&mut self.scalar_data[s_index]) }
    }

    pub fn get_scalar_standard_mut(&mut self, index: GlobalIndex) -> &mut [f32; 3] {
        self.get_scalar_mut(index.to_usize())
    }

    pub fn convert_to_fixed_point(&mut self) {
        //convert position vectors to fpi24_7 format
        self.convert_floating_vector_to_fixed::<fpi24_7>(GlobalIndex::Position);
        //convert normal and uv vectors to fpi3_28 format
        self.convert_floating_vector_to_fixed::<fpi3_28>(GlobalIndex::Normal);
        self.convert_floating_vector_to_fixed::<fpi3_28>(GlobalIndex::UV);
    }

    fn convert_floating_vector_to_fixed<T>(&mut self, global_index: GlobalIndex)
    where
        T: Copy + cmp::PartialEq + Sized + fmt::Debug + ScalarConvert<f32> + 'static,
    {
        let index = global_index.to_usize();

        for vec_index in 0..3 {
            for comp_index in 0..4 {
                //converted from f32 to fixed point
                let converted_comp: T = {
                    let vec_attrib = self.get_mut(index);
                    T::convert(vec_attrib[vec_index][comp_index])
                };
                unsafe {
                    let fixed_vec: &mut Vector4<T> = mem::transmute(
                        self.data
                            .as_mut_ptr()
                            .offset((3 * index + vec_index) as isize),
                    );
                    fixed_vec[comp_index] = converted_comp;
                }
            }
        }
    }
}

pub trait Graphics: Buffer2D<u32> + Sized
where
    Self: Index<(u32, u32)> + IndexMut<(u32, u32), Output = u32>,
{
    ///brute force implemtation of distance transform (compute the distance squared so that we dont need to use floating point operations)
    fn distance_transform_naive(&mut self) -> RowBuffer2D<u32> {
        const INF: u32 = 1 << 25;

        let w = self.width();
        let h = self.height();
        let mut dist_field = RowBuffer2D::new();
        dist_field.allocate(w, h, INF);

        for y0 in 0..h {
            for x0 in 0..w {
                for y1 in 0..h {
                    for x1 in 0..w {
                        //compute distance between (x0,y0) and (x1,y1)
                        let dx = x1 as i32 - x0 as i32;
                        let dy = y1 as i32 - y0 as i32;
                        let dist = dx * dx + dy * dy;
                        //fetch current distance
                        let cur_dist = dist_field[(y0, x0)];
                        //check if (x1,y1) is actually in bitmap
                        let y1p = if self.get((y1, x1)) > 0 { 0 } else { INF };
                        //update distance transformation
                        dist_field[(y0, x0)] = cur_dist.min(dist as u32 + y1p);
                    }
                }
            }
        }
        dist_field
    }

    ///converts a distance field to grey scale
    fn dist_to_grey_scale<T: Buffer2D<u32>>(&mut self, dest: &mut T) {
        let transform = self;
        let diagonal =
            255.0 / (transform.height().pow(2) as f32 + transform.width().pow(2) as f32).sqrt();
        for i in 0..transform.height() {
            for j in 0..transform.width() {
                let dist = ((transform.get((i, j)) as f32).sqrt() * diagonal) as u32 & 0xFF;
                let color = dist | dist << 8 | dist << 16 | dist << 24;
                dest.set((i, j), color);
            }
        }
    }

    // /// does what render_image(..) does but with fixed point math. This function may yield bad results
    // fn render_image_approx<T, S>(&mut self, dest: &mut T, mut bounds: AABB2D<u32>, sampler: S)
    // where
    //     T: Buffer2D<u32>,
    //     S: Sampler2D<fpi15_16, u32>,
    // {
    //     let cur_rect: Rect2D<u32> = Rect2D::new(0, 0, self.width(), self.height());
    //     bounds.clip_box(0, dest.width(), 0, dest.height());
    //     let dst_rect = bounds.to_rect();

    //     let scale_x =
    //         fpi15_16::convert(dst_rect.w) / fpi15_16::convert(fpi15_16::convert(cur_rect.w));
    //     let scale_y =
    //         fpi15_16::convert(dst_rect.h) / fpi15_16::convert(fpi15_16::convert(cur_rect.h));
    //     let w = dst_rect.w;
    //     let h = dst_rect.h;

    //     for y in 0..h {
    //         for x in 0..w {
    //             let sample_pos = (
    //                 fpi15_16::convert(x) / scale_x,
    //                 fpi15_16::convert(y) / scale_y,
    //             );
    //             let sample = sampler.interpolate(sample_pos.0, sample_pos.1, self);
    //             dest.set((dst_rect.y + y, dst_rect.x + x), sample);
    //         }
    //     }
    // }

    /// renders and image to dest buffer using nearest neighbour interpolation
    fn render_image<T: Buffer2D<u32>, S: Sampler2D<f32, u32>>(
        &self,
        dest: &mut T,
        mut bounds: AABB2D<i32>,
        sampler: S,
    ) {
        let cur_rect: Rect2D<u32> = Rect2D::new(0, 0, self.width(), self.height());
        bounds.clip_box(0, dest.width() as i32, 0, dest.height() as i32);
        let dst_rect = bounds.to_rect();

        let scale_x = cur_rect.w as f32 / dst_rect.w as f32;
        let scale_y = cur_rect.h as f32 / dst_rect.h as f32;
        let w = dst_rect.w as f32;
        let h = dst_rect.h as f32;
        for y in 0..h as u32 {
            for x in 0..w as u32 {
                let sample_pos = ((x as f32 * scale_x), (y as f32 * scale_y));
                let sample = sampler.interpolate(sample_pos.0, sample_pos.1, self);
                dest.set((dst_rect.y as u32 + y, dst_rect.x as u32 + x), sample);
            }
        }
    }

    /// Simple line drawing algorithm using (y=mx+b )
    fn draw_line(&mut self, a: Vector2<f32>, b: Vector2<f32>, color: u32) {
        let mut aabb = AABB2D::new();
        aabb.br.x = a.x.max(b.x);
        aabb.br.y = a.y.max(b.y);
        aabb.tl.x = a.x.min(b.x);
        aabb.tl.y = a.y.min(b.y);
        aabb.clip_box(0.0, self.width() as f32, 0.0, self.height() as f32);
        
        //bad names 
        let buf_width = self.width() as i32 - 1; 
        let buf_height = self.height() as i32 -1; 

        let rect = Rect2D::<i32>::from(aabb.to_rect());

        let dx = (b.x - a.x) as i32;
        let dy = (b.y - a.y) as i32;

        if rect.w > rect.h{
            if dx != 0 {
                let b = a.y as i32 - (dy * a.x as i32) / dx;
                for x_off in 0..rect.w {
                    let x = rect.x + x_off;
                    let mut y = (dy * x) / dx + b;
                    
                    y = y.max(0).min(buf_height as i32);

                    self[(y as u32, x as u32)] = color;
                }
            }
        } else {
            if dy != 0{
                let b = a.x as i32 - (dx * a.y as i32) / dy;
                for y_off in 0..rect.h {
                    let y = rect.y + y_off;
                    let mut x = (dx * y) / dy + b;
                    x = x.max(0).min(buf_width);

                    self[(y as u32, x as u32)] = color;
                }
            }
        }
    }

    fn draw_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32) {
        let mut aabb: AABB2D<i32> = AABB2D::new();
        let buffer = self;

        aabb.tl = Vector2::new(x, y);
        aabb.br = Vector2::new(x + w, y + h);
        aabb.clip_box(0, buffer.width() as i32, 0, buffer.height() as i32);
        let rect = aabb.to_rect();
        let color_block = unsafe { _mm_loadu_si128(transmute([color; 4].as_ptr())) };
        for i_off in 0..rect.h {
            let mut j_off = 0;
            while j_off + 4 < rect.w {
                let i = (rect.y + i_off) as u32;
                let j = (rect.x + j_off) as u32;
                let index = buffer.calc_index(i, j);
                let buffer_vec = buffer.as_vec_ref_mut();
                unsafe {
                    let pixel_ref = buffer_vec.get_unchecked_mut(index);
                    _mm_storeu_si128(transmute(pixel_ref), color_block);
                }
                j_off += 4;
            }
            for j_off_prime in (j_off - 4).max(0)..rect.w {
                let i = (rect.y + i_off) as u32;
                let j = (rect.x + j_off_prime) as u32;
                let index = buffer.calc_index(i, j);
                let buffer_vec = buffer.as_vec_ref_mut();
                unsafe {
                    *buffer_vec.get_unchecked_mut(index) = color;
                }
            }
        }
    }

    // /// Simpler draw routine (less optimizations)
    // fn draw_triangle<'a,'b,F>(&'a mut self, attribs:&VertAttribs,globals:&ShaderGlobals,buffers:&'a mut OutputBuffers,frag_shader:F)
    // where
    //     'a:'b,
    //     F: FnOnce(&(u32,u32),&Vector4<f32>,&VertAttribs,&ShaderGlobals,&mut Self,&mut OutputBuffers)->(u16,u32) + Copy
    // {
    //     const TILE_W:i32= 32;
    //     const TILE_H:i32= 32;
    //     const TILE_RAD_SQRD:i32 = (TILE_H/2)*(TILE_H/2) + (TILE_W/2)*(TILE_W/2);

    //     let mut aabb: AABB2D<f32> = AABB2D::new();
    //     let mut points = *attribs.get(0);

    //     // truncate vertex coordinates to the nearest subpixel (0.25)
    //     for pos in points.iter_mut(){
    //         pos.x = pos.x.floor() + (pos.x.fract()/0.5).floor()*0.5;
    //         pos.y = pos.y.floor() + (pos.y.fract()/0.5).floor()*0.5;
    //     }

    //     //compute bounding box
    //     aabb.update_aabb(&points);
    //     aabb.clip_box(0., self.width() as f32, 0., self.height() as f32);

    //     let rect_x = aabb.tl.x.floor() as i32;
    //     let rect_y = aabb.tl.y.floor() as i32;
    //     let rect_w = aabb.width() as i32;
    //     let rect_h = aabb.height() as i32;

    //     //compute triangle coeficients(only needs to be computed once per triangle)
    //     let triangle = Precomputed2DTriangleSDF::new(&points);

    //     let max_x = (aabb.br.x.ceil()).min(self.width() as f32 -1.) as i32;
    //     let max_y = (aabb.br.y.ceil()).min(self.height() as f32 -1.) as i32 ;

    //     let (area,_) = edge_function(points[0],points[1],points[2]);
    //     let inv_area = 1.0/area;
    //     let f_abg = area; //(area is signed)

    //     let off_screen_point = Vector4::new(-1.,-1.,0.,1.);

    //     let (bias01,_) = edge_function(points[0], points[1], off_screen_point);
    //     let (bias12,_) = edge_function(points[1], points[2], off_screen_point);
    //     let (bias20,_) = edge_function(points[2], points[0], off_screen_point);

    //     let cond12 = f_abg*bias12 > 0.;
    //     let cond20 = f_abg*bias20 > 0.;
    //     let cond01 = f_abg*bias01 > 0.;

    //     //compute inv depths of the triangle
    //     let depths_list = attribs.get_scalar(0);
    //     let depths = Vector4::new(
    //         1.0/depths_list[0],
    //         1.0/depths_list[1],
    //         1.0/depths_list[2],
    //         0.0,
    //     );

    //     let (mut px, mut py) = (rect_x,rect_y);

    //     let p0 = Vector4::new(rect_x as f32+0.5, rect_y as f32+0.5,0.0,1.0);
    //     let (f01,EdgeWeights{ a:a01,b:b01,..}) = edge_function(points[0],points[1],p0);
    //     let (f12,EdgeWeights{ a:a12,b:b12,..}) = edge_function(points[1],points[2],p0);
    //     let (f20,EdgeWeights{ a:a20,b:b20,..}) = edge_function(points[2],points[0],p0);
    //     #[allow(unused_assignments)]
    //     let (mut f,mut fr) = (Vector4::zeros(),Vector4::new(f12,f20,f01,0.)*inv_area);
    //     let mut weights;
    //     let delta_x = Vector4::new(a12,a20,a01,0.)*inv_area;
    //     let delta_y = Vector4::new(b12,b20,b01,0.)*inv_area;

    //     let dx2 = delta_x*0.5;
    //     let dy2 = delta_y*0.5;
    //     let bary_disps = [ dx2, -dx2, dy2, -dy2, Vector4::zeros()];

    //     // 'u' and 'v' are the coordinates in the unit square for the
    //     // bilinear interpolation (not related to the 'uv' in texturemapping)
    //     let (mut u , mut v) = (0.0,0.0);

    //     const E:f32 = 0.001;
    //     let edge0 = points[2] - points[1];
    //     let edge1 = points[0] - points[2];
    //     let edge2 = points[1] - points[0];

    //     let w0_cond = edge0.y.abs() < E  && edge0.x > 0.  || edge0.y > 0.;
    //     let w1_cond = edge1.y.abs() < E  && edge1.x > 0.  || edge1.y > 0.;
    //     let w2_cond = edge2.y.abs() < E  && edge2.x > 0.  || edge2.y > 0.;

    //     while py < max_y{
    //         f = fr;
    //         while px < max_x{
    //             weights = f;
    //             let mask = (weights.x.to_bits() | weights.y.to_bits() | weights.z.to_bits())>>31;
    //             let tm = unsafe{  TRIANGLE_MASK } ;

    //             //if SUB_STEPS==1 then sub_pixel is literally just the pixel position (px,py)
    //             let sub_pixel = (
    //                 py.max(0) as u32,
    //                 px.max(0) as u32,
    //             );

    //             let overlaps = |weights:&Vector4<f32>|{
    //                 let mut overlaps = true;
    //                 overlaps&= if weights[0].abs() < E { w0_cond  }else{  weights[0] > 0. };
    //                 overlaps&= if weights[1].abs() < E { w1_cond  }else{  weights[1] > 0. };
    //                 overlaps&= if weights[2].abs() < E { w2_cond  }else{  weights[2] > 0. };
    //                 overlaps
    //             };

    //             for d in &bary_disps{
    //                 weights = f+d;
    //                 if tm || overlaps(&weights)  {
    //                     let world_z = 1.0/weights.dot(&depths);
    //                     weights*=world_z;
    //                     weights[0]*=depths[0];
    //                     weights[1]*=depths[1];
    //                     weights[2]*=depths[2];
    //                     frag_shader(&sub_pixel,&weights,&attribs,&globals,self,buffers);
    //                     break;
    //                 }
    //             }

    //             //horizontal increments
    //             px+=1;
    //             f+=delta_x;
    //         }

    //         //horizontal resets
    //         px=rect_x;
    //         u = 0.;

    //         //vertical increments
    //         py+=1;
    //         fr+=delta_y;
    //     }
    // }

    ///Draws triangle with following optimizations:
    ///1. tiling optimization to skip empty space        [x]
    ///2. iterative barycentric optimization             [x]
    ///3. fill rules that appear to be working           [ ]
    ///4. tiling optimization to reduce bandwith         [ ]
    ///5. fast perspective correct barycentric weights   [x]
    ///6. a custom SIMD implementation for x86 and ARM?  [ ]
    #[no_mangle]
    fn draw_triangle<F>(
        &mut self,
        attribs: &TriangleData,
        globals: &ShaderGlobals,
        buffers: &mut OutputBuffers,
        frag_shader: F,
    ) where
        F: FnOnce(
                &(u32, u32),
                &Vector4<f32>,
                &TriangleData,
                &ShaderGlobals,
                &mut Self,
                &mut OutputBuffers,
            ) -> (u16, u32)
            + Copy,
    {
        const TILE_W: i32 = 4;
        const TILE_H: i32 = 4;
        const TILE_RAD_SQRD: i32 = TILE_W * TILE_W / 4 + TILE_H * TILE_H / 4;

        let mut aabb: AABB2D<f32> = AABB2D::new();
        let mut points = *attribs.get(0);

        //compute bounding box and clip it so it stays on screen
        aabb.update_aabb(&points);
        aabb.clip_box(0., self.width() as f32, 0., self.height() as f32);

        let rect_x = aabb.tl.x.floor() as i32;
        let rect_y = aabb.tl.y.floor() as i32;
        let rect_w = aabb.width() as i32;
        let rect_h = aabb.height() as i32;

        if rect_w <= 0 || rect_h <= 0 {
            return;
        }

        //compute triangle coeficients(only needs to be computed once per triangle)
        let triangle = Precomputed2DTriangleSDF::new(&points);

        let max_x = (aabb.br.x.ceil()).min(self.width() as f32 - 1.) as i32;
        let max_y = (aabb.br.y.ceil()).min(self.height() as f32 - 1.) as i32;

        let (area, _) = edge_function(points[0], points[1], points[2]);
        let inv_area = 1.0 / area;

        //compute inv depths of the triangle
        let depths_list = attribs.get_scalar(0);
        let depths = Vector4::new(
            1.0 / depths_list[0],
            1.0 / depths_list[1],
            1.0 / depths_list[2],
            0.0,
        );

        //compute barycentric coord for the top left aabb of the triangle
        let p0 = Vector4::new(rect_x as f32 + 0.5, rect_y as f32 + 0.5, 0.0, 1.0);
        let (f01, EdgeWeights { a: a01, b: b01, .. }) = edge_function(points[0], points[1], p0);
        let (f12, EdgeWeights { a: a12, b: b12, .. }) = edge_function(points[1], points[2], p0);
        let (f20, EdgeWeights { a: a20, b: b20, .. }) = edge_function(points[2], points[0], p0);

        let fr = Vector4::new(f12, f20, f01, 0.) * inv_area;
        let delta_x = Vector4::new(a12, a20, a01, 0.) * inv_area;
        let delta_y = Vector4::new(b12, b20, b01, 0.) * inv_area;

        if fr[0].abs() > 2048. || fr[1].abs() > 2048. || fr[2].abs() > 2048. {
            // println!("degenerate triangle detected!");
            // println!("[x:{},y:{},w:{},h:{},area:{}]",rect_x,rect_y,rect_w,rect_h,area);
            // for k in 0..3{
            //     println!("{}",points[k]);
            // }
            // panic!("");
            return;
        }

        use crate::MAX_FR;
        use crate::MIN_FR;
        unsafe {
            for k in 0..3 {
                MIN_FR[k] = MIN_FR[k].min(fr[k]);
                MAX_FR[k] = MAX_FR[k].max(fr[k]);
            }
            MIN_FR[3] = MIN_FR[3].min(inv_area);
            MAX_FR[3] = MAX_FR[3].max(inv_area);
        }

        for ty in 0..(rect_h / TILE_H) + 1 {
            for tx in 0..(rect_w / TILE_W) + 1 {
                let tx0 = tx * TILE_W + rect_x;
                let ty0 = ty * TILE_H + rect_y;
                let tw = (tx0 + TILE_W).min(max_x) - tx0;
                let th = (ty0 + TILE_H).min(max_y) - ty0;
                let tile_dim = Rect2D {
                    x: tx0,
                    y: ty0,
                    w: tw,
                    h: th,
                };

                //only draw cells that intersect triangle
                if cell_overlaps_triangle(tx0, ty0, tw, th, &triangle) {
                    let off_x = tx * TILE_W;
                    let off_y = ty * TILE_H;
                    let mut tl_tile_bary = fr + delta_x * (off_x as f32) + delta_y * (off_y as f32);

                    Self::fill_triangle_tile(
                        &mut points,
                        &depths,
                        tile_dim,
                        &tl_tile_bary,
                        &delta_x,
                        &delta_y,
                        attribs,
                        globals,
                        self,
                        buffers,
                        frag_shader,
                    );
                }
            }
        }
    }

    #[no_mangle]
    fn fill_triangle_tile<F>(
        points: &mut [Vector4<f32>; 3],
        depths: &Vector4<f32>,
        tile_dim: Rect2D<i32>,
        fr: &Vector4<f32>,
        delta_x: &Vector4<f32>,
        delta_y: &Vector4<f32>,
        attribs: &TriangleData,
        globals: &ShaderGlobals,
        main_buffer: &mut Self,
        buffers: &mut OutputBuffers,
        frag_shader: F,
    ) where
        F: FnOnce(
                &(u32, u32),
                &Vector4<f32>,
                &TriangleData,
                &ShaderGlobals,
                &mut Self,
                &mut OutputBuffers,
            ) -> (u16, u32)
            + Copy,
    {
        let tx0 = tile_dim.x as u32;
        let ty0 = tile_dim.y as u32;
        let tw = tile_dim.w;
        let th = tile_dim.h;

        let (mut x, mut y) = (0, 0);
        let (mut px, mut py) = (tx0, ty0);
        let (mut f, mut fr) = (Vector4::zeros(), *fr);

        //Here I'm calculating the barycentric weights for each corner of the "tile/cell"
        let f_tl = fr;
        let f_tr = fr + delta_x * (tw as f32);
        let f_bl = fr + delta_y * (th as f32);
        let f_br = fr + delta_x * (tw as f32) + delta_y * (th as f32);

        //calculate exact worldspace z for each corner of the tile
        let world_z_tl = 1.0 / f_tl.dot(&depths);
        let world_z_tr = 1.0 / f_tr.dot(&depths);
        let world_z_bl = 1.0 / f_bl.dot(&depths);
        let world_z_br = 1.0 / f_br.dot(&depths);
        let grid_table = [world_z_tl, world_z_tr, world_z_bl, world_z_br];

        // 'u' and 'v' are the coordinates in the unit square for the
        // bilinear interpolation (not related to the 'uv' in texturemapping)
        let (mut u, mut v) = (0.0, 0.0);

        //calculate stepsize for linear interpolations in
        // horizontal and vertical directions
        let (u_step, v_step) = (1.0 / tw as f32, 1.0 / th as f32);

        //compute displacements for subpixel sampling
        let subx: Vector4<f32> = delta_x * 0.25;
        let suby: Vector4<f32> = delta_y * 0.25;
        let bary_disp_list = [subx + suby, -subx + suby, -suby - subx, -suby + subx];

        //define linear interpolation here
        let lerp = |a, b, t: f32| (b - a) * t + a;

        //Calculating  '1/x' per pixel is not really an option for me, so
        //in order to avoid this,I calculate 1/x at the four tile corners and
        //do bilinear interpolation to avoid doing a 1/x per pixel
        let calc_approx_z = |u, v| {
            lerp(
                lerp(world_z_tl, world_z_tr, u),
                lerp(world_z_bl, world_z_br, u),
                v,
            )
        };
        let mut shader_color;
        let mut shader_depth;

        let depth_buffer =
            unsafe { &mut *buffers.get_scalar_buffer(GlobalIndex::PrimaryDepthBuffer) };

        while y < th {
            f = fr;

            while x < tw && (tw - x) % 3 > 0 {
                //approximate worldspace z for current pixel with bilinear interpoaltion
                //this approximation is not accurate enough for sub-pixel sampling
                let world_z = calc_approx_z(u, v); //sampler::cached_bilinear_f32(u,v,&grid_table);
                let pixel_index = (py, px);
                let _tm = unsafe { TRIANGLE_MASK };
                let mut weights;

                //preform subpixel sampling
                for sub_disp in bary_disp_list.iter() {
                    weights = f + sub_disp;
                    let mask = (weights.x.to_bits() | weights.y.to_bits() | weights.z.to_bits())
                        & (1 << 31);

                    if mask == 0 {
                        // main_buffer.set(pixel_index,0x00ff00ff);
                        //compute  perspective correct barycentric weights( approximately w/ bilerp)
                        weights *= world_z; //1.0/depths.dot(&weights);
                        weights[0] *= depths[0];
                        weights[1] *= depths[1];
                        weights[2] *= depths[2];

                        frag_shader(
                            &pixel_index,
                            &weights,
                            &attribs,
                            &globals,
                            main_buffer,
                            buffers,
                        );
                        break;
                    }
                }

                //horizontal increments
                px += 1;
                x += 1;
                f += delta_x * 1.0;
                u += u_step * 1.0;
            }

            while x < tw {
                let world_z = calc_approx_z(u, v);
                let mut pixel_index = (py, px);
                let mut weights;
                let mut mask = 1;
                weights = f;
                //compute first pixel in block to intersect triangle
                for _ in 0..3 {
                    mask = (weights.x.to_bits() | weights.y.to_bits() | weights.z.to_bits()) as i32
                        >> 31;
                    if mask == 0 {
                        break;
                    }
                    weights += delta_x;
                    pixel_index.1 += 1;
                }

                if mask == 0 {
                    // main_buffer.set(pixel_index,0x00ff00ff);
                    //compute  perspective correct barycentric weights( approximately w/ bilerp)
                    weights *= world_z;
                    weights[0] *= depths[0];
                    weights[1] *= depths[1];
                    weights[2] *= depths[2];
                    let (d, c) = frag_shader(
                        &pixel_index,
                        &weights,
                        &attribs,
                        &globals,
                        main_buffer,
                        buffers,
                    );
                    shader_color = c;
                    shader_depth = d;

                    for k in 0..3 {
                        main_buffer.set((py + 0, px + k), shader_color);
                        depth_buffer.set((py + 0, px + k), shader_depth);
                    }
                }
                //horizontal increments
                px += 3;
                x += 3;
                f += delta_x * 3.0;
                u += u_step * 3.0;
            }

            //horizontal resets
            x = 0;
            px = tx0;
            u = 0.;

            //vertical increments
            py += 1;
            y += 1;
            fr += delta_y * 1.0;
            v += v_step * 1.0;
        }
    }
}

impl Graphics for RowBuffer2D<u32> {}
impl Graphics for TiledBuffer2D<u32> {}

pub trait ColorConvertUtils {
    fn to_argb8888(&self) -> u32;
}

impl ColorConvertUtils for Vector4<f32> {
    fn to_argb8888(&self) -> u32 {
        let color = self;
        ((color.w * 255.0) as u32) << 24
            | ((color.x * 255.0) as u32) << 16
            | ((color.y * 255.0) as u32) << 8
            | ((color.z * 255.0) as u32)
    }
}

// #[inline]
fn custom_frag_shader<BUF: Buffer2D<u32>>(
    pixel_pos: &(u32, u32),
    weights: &Vector4<f32>,
    attr: &TriangleData,
    glbs: &ShaderGlobals,
    color_buffer: &mut BUF,
    aux_buffers: &mut OutputBuffers,
) {
    // //grab depth attributes
    // let depths = attr.get_scalar_standard(GlobalIndex::VertexDepths);
    // //grab depth_buffer
    // let depth_buffer =  unsafe{ &mut *aux_buffers.get_scalar_buffer(GlobalIndex::PrimaryDepthBuffer)};

    // //perspective correct depth calculation
    // let current_depth = depth_buffer.get(pixel_pos) ;
    // let new_depth = ((depths[0]  * weights.x + depths[1]  * weights.y + depths[2] * weights.z).abs()*(255.0/100.0)) as u8;

    // //branchless depth calculation

    // let depth_mask:u32 = unsafe{ std::mem::transmute((new_depth as i32 - current_depth as i32) >> 31)} ;
    // let final_depth_bits = !depth_mask  & current_depth as u32| depth_mask & new_depth as u32 ;
    // let final_depth = final_depth_bits ;

    // depth_buffer.set(pixel_pos, final_depth as u8 );

    //get positions
    // let pos_lst = attr.get(0);
    // let position = pos_lst[0] * weights[0] + pos_lst[1] * weights[1] + pos_lst[2] * weights[2];

    //get normals
    // let normals = attr.get(1);
    // let normal = normals[0] * weights[0] + normals[1] * weights[1] + normals[2] * weights[2];

    // if depth_mask > 0{
    // linearly interpolate uvs
    let uvs = attr.get_standard(GlobalIndex::UV);
    let uv = uvs[0] * weights[0] + uvs[1] * weights[1] + uvs[2] * weights[2];

    //grab texture
    let texture = unsafe { &mut *aux_buffers.texture_buffers[0].unwrap() };
    let new_color =
        <NearestSampler2D as Sampler2D<f32, u32>>::uv_map(&NearestSampler2D, uv, texture);
    color_buffer.set(*pixel_pos, new_color);
    // }

    // let current_color = color_buffer.get(pixel_pos);
    // let new_color = <NearestSampler2D as Sampler2D<f32, u32>>::uv_map(&NearestSampler2D, uv, texture);
    // let final_color = !depth_mask & current_color | depth_mask & new_color;
    // color_buffer.set(pixel_pos, final_color);
}
