use std::ops::{Index,IndexMut};
use std::fmt;


use super::super::{
    quick_log2_unchecked,
    is_pow2,
};

use super::{
    BufferConvert,
    RasterDimension,
    Buffer2D,
};

impl<T: Default + Copy + std::ops::Mul> RasterDimension<T> {
    pub fn new() -> RasterDimension<T> {
        RasterDimension {
            cols: T::default(),
            rows: T::default(),
        }
    }
    pub fn set(&mut self, cols: T, rows: T) {
        self.cols = cols;
        self.rows = rows;
    }
    pub fn area(&self) -> <T as std::ops::Mul>::Output {
        self.cols * self.rows
    }
}

impl<T: Default + Copy> Default for RasterDimension<T> {
    fn default() -> Self {
        RasterDimension {
            cols: T::default(),
            rows: T::default(),
        }
    }
}

pub struct RowBuffer2D<T> {
    pub dim: RasterDimension<u32>,
    pub data: Vec<T>,
}

impl<T: Copy + Default> RowBuffer2D<T> {
    pub fn new() -> RowBuffer2D<T> {
        RowBuffer2D {
            dim: RasterDimension::default(),
            data: Vec::new(),
        }
    }

    pub fn allocate(&mut self, w: u32, h: u32, base: T) {
        self.dim.set(w, h);
        self.data = (0..w * h).map(|_| base).collect();
    }

    pub fn allocate_dim(&mut self, dim: RasterDimension<u32>, base: T)  {
        self.allocate(dim.cols, dim.rows, base)
    }
}

impl<T: Copy + fmt::Display + Default  > Buffer2D<T> for RowBuffer2D<T> {
    fn width(&self) -> u32 {
        self.dim.cols
    }
    fn height(&self) -> u32 {
        self.dim.rows
    }

    fn resize(&mut self, w:i32,h:i32){
        self.dim.cols = w.max(0) as u32 ; 
        self.dim.rows = h.max(0) as u32 ;
        self.data.resize((w*h) as usize,T::default());
    }

    fn as_vec_ref(&self)->&Vec<T>{
        &self.data
    }
    fn as_vec_ref_mut(&mut self)->&mut Vec<T>{
        &mut self.data
    }

    fn calc_index(&self, i: u32, j: u32) -> usize {
        (i * self.width() + j) as usize
    }
}

impl<T: Copy + fmt::Display +Default > Index<(u32, u32)> for RowBuffer2D<T> {
    type Output = T;
    fn index(&self, index_tup: (u32, u32)) -> &Self::Output {
        let index = self.calc_index(index_tup.0, index_tup.1);
        unsafe { self.as_vec_ref().get_unchecked(index) }
    }
}

impl<T: Copy + fmt::Display +Default > std::ops::IndexMut<(u32, u32)> for RowBuffer2D<T> {
    fn index_mut(&mut self, index_tup: (u32, u32)) -> &mut Self::Output {
        let index = self.calc_index(index_tup.0, index_tup.1);
        unsafe { self.as_vec_ref_mut().get_unchecked_mut(index) }
    }
}




pub struct TiledBuffer2D<T> {
    pub external_dim: RasterDimension<u32>,
    pub tile_dim: RasterDimension<u32>,
    pub block_dim: RasterDimension<u32>,
    pub block_log: RasterDimension<u32>, 
    pub internal_buffer: RowBuffer2D<T>,
}

impl<T: Copy + Default> TiledBuffer2D<T> {
    pub fn new() -> TiledBuffer2D<T> {
        TiledBuffer2D {
            external_dim: RasterDimension::default(),
            tile_dim: RasterDimension::default(),
            block_dim: RasterDimension::default(),
            block_log: RasterDimension::default(),
            internal_buffer: RowBuffer2D::<T>::new(),
        }
    }

    pub fn allocate(
        mut self,
        w: u32,
        h: u32,
        block_cols: u32,
        block_rows: u32,
    ) -> Result<TiledBuffer2D<T>, &'static str> {
        self.external_dim.set(w, h);
        self.block_dim.set(block_cols, block_rows);

        if is_pow2(self.block_dim.cols) == false || is_pow2(self.block_dim.rows) == false {
            return Err("block_dim should be a power of 2");
        }
        let dim = self.calculate_internal_dim();
        self.internal_buffer.allocate_dim(dim, T::default());

        self.block_log.cols = unsafe{ quick_log2_unchecked(self.block_dim.cols)};
        self.block_log.rows = unsafe{ quick_log2_unchecked(self.block_dim.rows)};
        
        Ok(self)
    }

    pub fn print_debug_info(&self){
        println!(
            "external bounds: {}x{}\ninternal bounds: {}x{}\nUtilization:{}",
            self.external_dim.cols,
            self.external_dim.rows,
            self.internal_buffer.dim.cols,
            self.internal_buffer.dim.rows,
            (100.0 * self.external_dim.area() as f32 / self.internal_buffer.dim.area() as f32)
                .round()
        );
    }
    fn calculate_internal_dim(&mut self) -> RasterDimension<u32> {
        let mut intern_dim = RasterDimension::default();
        //determines if padding is neccessary
        let col_padding = if self.external_dim.cols % self.block_dim.cols == 0 { 0 } else { 1 };
        let row_padding = if self.external_dim.rows % self.block_dim.rows == 0 { 0 } else { 1 };
        
        //calculates the internal texture size
        intern_dim.set(
            (self.external_dim.cols / self.block_dim.cols + col_padding) * self.block_dim.cols,
            (self.external_dim.rows / self.block_dim.rows + row_padding) * self.block_dim.rows,
        );
        
        //calculates tiles required to fit in the blocks
        self.tile_dim.cols = intern_dim.cols / self.block_dim.cols;
        self.tile_dim.rows = intern_dim.rows / self.block_dim.rows;

        intern_dim
    }
    pub fn internal_len(&self)->u32{
        self.internal_buffer.dim.area()
    }
}

impl<T: Copy + fmt::Display >  Buffer2D<T> for TiledBuffer2D<T> {
    fn width(&self) -> u32 {
        self.external_dim.cols
    }

    fn height(&self) -> u32 {
        self.external_dim.rows
    }

    fn resize(&mut self, w:i32,h:i32){
        panic!("resize(..) not implemented! for TiledBuffer<T>")
    }

    fn as_vec_ref(&self)->&Vec<T>{
        &self.internal_buffer.data
    }

    fn as_vec_ref_mut(&mut self)->&mut Vec<T>{
        &mut self.internal_buffer.data
    }

    fn calc_index(&self, i: u32, j: u32) -> usize {
        let block_log_rows = self.block_log.rows;//unsafe { quick_log2_unchecked(self.block_dim.rows) };
        let block_log_cols = self.block_log.cols;//unsafe { quick_log2_unchecked(self.block_dim.cols) };
        let block_index = (i & (self.block_dim.rows - 1)) * self.block_dim.cols + (j & (self.block_dim.cols - 1));
        let tile_offset = (i >> block_log_rows) * self.tile_dim.cols + (j >> block_log_cols);
        let index = block_index + tile_offset * (self.block_dim.cols * self.block_dim.rows);
        index as usize
    }
}

impl<T: Copy +Sized+fmt::Display > Index<(u32, u32)> for TiledBuffer2D<T> {
    type Output = T;
    fn index(&self, index_tup: (u32, u32)) -> &Self::Output {
        let index = self.calc_index(index_tup.0, index_tup.1);
        unsafe { self.as_vec_ref().get_unchecked(index) }
    }
}

impl<T: Copy + Sized+fmt::Display > IndexMut<(u32, u32)> for TiledBuffer2D<T> {
    fn index_mut(&mut self, index_tup: (u32, u32)) -> &mut Self::Output {
        let index = self.calc_index(index_tup.0, index_tup.1);
        unsafe { self.as_vec_ref_mut().get_unchecked_mut(index) }
    }
}

pub mod buffer_tests{
    use super::*;
    
    #[test]
    pub fn tile2d_test(){
        let mut index_list = Vec::new(); 
        for w in 1..=32u32{
            for h in w..=32u32{
                for k in 1..16{
                    let size = 1<<k;
                    if size*size > w*h{ continue }
                    assert_eq!((true,w,h,size,size),tile2d_index_sanity_check(w, h, size ,size,&mut index_list));
                    index_list.clear();
                }
            }
        }
    }
    /// Checks if the index over the external bounds is within internal bounds or 
    /// to put another way: checks if index calculated points to valid memory 
    /// This test also checks if the index pideonholes. 
    pub fn tile2d_index_sanity_check(w:u32,h:u32,tile_w:u32,tile_h:u32,index_list:&mut Vec<usize>)->(bool,u32,u32,u32,u32){
        let is_unique = |index_list: &Vec<usize>| {
            let mut unique = true;
            for k in 1..index_list.len() as usize {
                if index_list[k] == index_list[k - 1] {
                    unique = false;
                    break;
                }
            }
            unique
        };

        let buffer: TiledBuffer2D<u32> = TiledBuffer2D::new().allocate(w, h, tile_w, tile_h).unwrap();
        for i in 0..buffer.height() {
            for j in 0..buffer.width() {
                let index = buffer.calc_index(i, j);
                index_list.push(index);
            }
        }
        index_list.sort_unstable();
        let result = is_unique(&index_list)  && (*index_list.last().unwrap() as u32) < buffer.internal_len();

        (result,w,h,tile_w,tile_h)
    }
}

impl<T> BufferConvert<TiledBuffer2D<T>, T> for RowBuffer2D<T> where T: Copy + Default + fmt::Display {}