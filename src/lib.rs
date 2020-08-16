#[macro_use]
extern crate lazy_static;


pub mod image; 
pub mod graphics;
pub mod simd_util; 
pub mod math_util; 
pub mod bounding; 
pub mod time_util; 
pub mod system; 
pub mod model; 
pub mod containers;
mod physics; 


const fn init_fr(val:f32)->[f32;4]{
    [val;4]
}
pub static mut MIN_FR:[f32;4] = init_fr(9999.9);
pub static mut MAX_FR:[f32;4] = init_fr(-9999.9);

#[inline]
pub fn is_pow2(x: u32) -> bool {
    x != 0 && x & (x - 1) == 0  
}
/// calculates binary log of x. The function unsafe because if x isnt a power of two
/// the function will return an incorrect answer
/// ***
/// More specifically it computes:\
/// log_2(x), assuming that x = 2^k , for some k >=0.

#[inline]
pub unsafe fn quick_log2_unchecked(x: u32) -> u32 {
    (x - 1).count_ones()
}

