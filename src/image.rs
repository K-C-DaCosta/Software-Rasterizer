pub mod buffer;
pub mod sampler;

use png;
use std::fmt;
use std::fs::File;
use std::mem::*;
use std::ops;
use std::ptr;
use std::slice::*;


pub trait Buffer2D<T: Copy + fmt::Display>: ops::Index<(u32, u32)> + ops::IndexMut<(u32, u32)> {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn resize(&mut self, w: i32, h: i32);
    fn calc_index(&self, i: u32, j: u32) -> usize;
    fn as_vec_ref(&self) -> &Vec<T>;
    fn as_vec_ref_mut(&mut self) -> &mut Vec<T>;

    fn fill(&mut self, fill_val: T) {
        for elem in self.as_vec_ref_mut() {
            *elem = fill_val;
        }
    }

    fn fill_zero(&mut self) {
        unsafe {
            ptr::write_bytes(self.as_vec_ref_mut().as_mut_ptr(), 0, self.len() as usize);
        }
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        let vec_ref = self.as_vec_ref_mut();
        let size = vec_ref.len();
        unsafe { from_raw_parts_mut(vec_ref.as_ptr() as *mut u8, size * size_of::<T>()) }
    }

    fn len(&self) -> u32 {
        self.height() * self.width()
    }

    fn get(&self, index2d: (u32, u32)) -> T {
        let index1d = self.calc_index(index2d.0, index2d.1);
        // let result = self.as_vec_ref().get(index1d).unwrap() ; // safe version but slow
        let result = unsafe { self.as_vec_ref().get_unchecked(index1d) };
        *result
    }

    fn set_all(&mut self,val:T){
        self.as_vec_ref_mut()
            .iter_mut()
            .for_each(|element| *element = val);
    }
    
    fn set(&mut self, index2d: (u32, u32), val: T) {
        let index1d = self.calc_index(index2d.0, index2d.1);
        unsafe {
            // *self.as_vec_ref_mut().get_mut(index1d).unwrap() = val; // safe version but slow
            *self.as_vec_ref_mut().get_unchecked_mut(index1d) = val;
        }
    }

    fn dampen(&mut self) {
        const SCALE_TABLE: [i32; 4] = [0x3f3f3f3f; 4];
        static mut K: u32 = 0;
        /*quick_dampen(...)*/
        {
            use super::simd_util::*;
            use std::arch::x86_64::*;
            let buffer = self;
            //loop through buffer and apply dampening ( 4 pixels at once per loop)
            unsafe {
                let scale = _mm_loadu_si128(transmute(SCALE_TABLE.as_ptr()));
                let mask = _mm_loadu_si128(transmute(BYTE_MASK.as_ptr()));
                let shift = _mm_loadu_si128(transmute(SHIFT_MASK.as_ptr()));
                K += 1;
                K &= 127;
                for k in ((K % 4) << 2..buffer.len()).step_by(16) {
                    let pixel = buffer.as_vec_ref_mut().as_mut_ptr().offset(k as isize);
                    let batch = _mm_loadu_si128(transmute(pixel));
                    let result = _mm_mul_damp2_i8(batch, scale, mask, shift);
                    _mm_storeu_si128(transmute(pixel), result);
                }
            }
        }
    }

    fn print(&self) {
        for j in 0..self.height() {
            for i in 0..self.width() {
                print!("[{:3}]", self.get((i, j)));
            }
            print!("\n");
        }
    }
}

pub trait BufferConvert<OUT, T>: Buffer2D<T>
where
    T: Copy + Default + fmt::Display ,
    OUT: Buffer2D<T> + ops::IndexMut<(u32,u32),Output=T>,
    Self:ops::Index<(u32,u32),Output=T>
{
    fn encode(&self, output: &mut OUT) {
        for i in 0..self.height() {
            for j in 0..self.width() {
                output[(i,j)]=self[(i,j)];
            }
        }
    }
}
pub struct RasterDimension<T> {
    pub cols: T,
    pub rows: T,
}

pub fn load_texture(path: &str) -> Option<buffer::RowBuffer2D<u32>> {
    use buffer::RowBuffer2D;
    use png::ColorType;
    use png::Transformations;

    match File::open(path) {
        Ok(file) => {
            let mut buffer: RowBuffer2D<u32> = RowBuffer2D::new();
            let decoder = png::Decoder::new(file);

            let (info, mut reader) = decoder.read_info().unwrap();

            println!(
                "width = {} height ={}\nline_size ={}\nsamples ={}",
                info.width,
                info.height,
                info.line_size,
                info.color_type.samples()
            );

            buffer.allocate(info.width as u32, info.height as u32, 0);

            let mut temp_buffer = vec![0; info.buffer_size()];

            match reader.next_frame(&mut temp_buffer[..]) {
                Ok(_) => {
                    match info.color_type {
                        ColorType::RGB => {
                            // println!("RGB");
                            //transfrom to argb888
                            for k in (0..temp_buffer.len()).step_by(3) {
                                let b = (temp_buffer[k + 2] as u32) << 0;
                                let g = (temp_buffer[k + 1] as u32) << 8;
                                let r = (temp_buffer[k + 0] as u32) << 16;
                                buffer.as_vec_ref_mut()[k / 3] = r | g | b;
                            }
                        }
                        ColorType::RGBA => {
                            // println!("RGBA");
                            //transfrom to argb888
                            for k in (0..temp_buffer.len()).step_by(4) {
                                let a = (temp_buffer[k + 3] as u32) << 24;
                                let b = (temp_buffer[k + 2] as u32) << 0;
                                let g = (temp_buffer[k + 1] as u32) << 8;
                                let r = (temp_buffer[k + 0] as u32) << 16;
                                buffer.as_vec_ref_mut()[k / 4] = a | r | g | b;
                            }
                        }
                        _ => return None,
                    }
                    Some(buffer)
                }
                Err(err) => {
                    println!("error: {}", err);
                    None
                }
            }
        }
        Err(_) => None,
    }
}

pub fn load_tiled_texture(path:&str,block_cols:u32,block_rows:u32)->Option<buffer::TiledBuffer2D<u32>>{
    let result = load_texture(path);
    
    if let Some(mut texture) = result{
        let mut tiled = buffer::TiledBuffer2D::<u32>::new().allocate(texture.width(), texture.height(), block_cols, block_rows).unwrap();
        texture.encode(&mut tiled);
        Some(tiled)
    } else{
        None
    }
}
