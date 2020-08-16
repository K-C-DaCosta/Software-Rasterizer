#![allow(dead_code)]

use std::mem::*;
use std::arch::x86_64::*; 
use super::image::{Buffer2D};

pub static BYTE_MASK:[u32;4] = [0x00ff00ff,0x00ff00ff,0x00ff00ff,0x00ff00ff];
pub static SHIFT_MASK:[u32;4] = [6,0,0,0];  


const fn mm_shuffle(a:i32,b:i32,c:i32,d:i32)->i32{
    

    a&0x3 | b&0x3 << 2 | c&3 << 4 | d&3 << 6
}

unsafe fn _mm_mul_i32(mut a:__m128i,mut b:__m128i)->__m128i{
    let mask = _mm_load_si128(transmute([!0,0,!0,0].as_mut_ptr()));
    let prod1 = _mm_and_si128(mask,_mm_mul_epu32(a,b)); 
    a = _mm_bsrli_si128(a,4); 
    b = _mm_bsrli_si128(b,4);
    let prod2 =  _mm_and_si128(mask,_mm_mul_epu32(  a,b)) ; 
    _mm_or_si128(prod1,_mm_bslli_si128(prod2,4))
}

#[inline]
unsafe fn _mm_mul_i8(mut a:__m128i,mut b : __m128i)->__m128i{
    let mask = _mm_load_si128(transmute([0x00ff00ff,0x00ff00ff,0x00ff00ff,0x00ff00ff].as_mut_ptr()));
    let prod1 = _mm_mullo_epi16(_mm_and_si128(mask,a),_mm_and_si128(mask,b));
    a = _mm_and_si128(mask,_mm_bsrli_si128(a,1)); 
    b = _mm_and_si128(mask,_mm_bsrli_si128(b,1)); 
    let prod2 = _mm_mullo_epi16(_mm_and_si128(mask,a),_mm_and_si128(mask,b));
    _mm_or_si128(prod1,_mm_bslli_si128(prod2,1))
}


#[inline]
pub unsafe fn _mm_mul_damp_i8(mut a:__m128i,mut b : __m128i)->__m128i{
    let mask  = _mm_load_si128(transmute(BYTE_MASK.as_ptr()));
    let shift = _mm_load_si128(transmute(SHIFT_MASK.as_ptr()));
    let mut prod1 = _mm_mullo_epi16(_mm_and_si128(mask,a),_mm_and_si128(mask,b));
    a = _mm_and_si128(mask,_mm_bsrli_si128(a,1)); 
    b = _mm_and_si128(mask,_mm_bsrli_si128(b,1)); 
    let mut prod2 = _mm_mullo_epi16(a,b);
    prod1  = _mm_sra_epi16( prod1 ,shift);
    prod2  = _mm_sra_epi16( prod2 ,shift);
    _mm_or_si128(prod1,_mm_bslli_si128(prod2,1))
}

#[inline]
pub unsafe fn _mm_mul_damp2_i8(mut a:__m128i,mut b : __m128i,mask:__m128i,shift:__m128i)->__m128i{
    let mut prod1 = _mm_mullo_epi16(_mm_and_si128(mask,a),_mm_and_si128(mask,b));
    a = _mm_and_si128(mask,_mm_bsrli_si128(a,1)); 
    b = _mm_and_si128(mask,_mm_bsrli_si128(b,1)); 
    let mut prod2 = _mm_mullo_epi16(a,b);
    prod1  = _mm_sra_epi16( prod1 ,shift);
    prod2  = _mm_sra_epi16( prod2 ,shift);
    _mm_or_si128(prod1,_mm_bslli_si128(prod2,1))
}


#[inline]
unsafe fn _mm_sra_epi8(mut a:__m128i,k:u32)->__m128i{
    let shift = _mm_loadu_si128(transmute([k,0,0,0].as_ptr()));
    let mask  = _mm_loadu_si128(transmute([0x00ff00ff,0x00ff00ff,0x00ff00ff,0x00ff00ff].as_ptr()));
    let part1 =_mm_sra_epi16( _mm_and_si128(mask,a) , shift);
    a =_mm_and_si128(mask,_mm_bsrli_si128(a,1)); 
    let part2 =_mm_sra_epi16(a, shift);  
    _mm_or_si128(part1,_mm_bslli_si128(part2,1))
}

#[allow(overflowing_literals)]
fn quick_dampen<T:Buffer2D<u32>>(buffer:&mut T){
    const SCALE_TABLE:[i32;4] = [0x3f3f3f3f;4];
    static mut  K:u32 = 0; 
    //loop through buffer and apply dampening ( 4 pixels at once per loop)
    unsafe{
        let scale = _mm_loadu_si128(transmute(SCALE_TABLE.as_ptr()));
        let mask  = _mm_load_si128(transmute(BYTE_MASK.as_ptr()));
        let shift = _mm_load_si128(transmute(SHIFT_MASK.as_ptr()));
        K+=1; 
        K&=127;
        for k in ( (K%4)<<2 .. buffer.len() ).step_by(16) {
            let pixel = buffer.as_vec_ref_mut().as_mut_ptr().offset(k as isize);
            let batch = _mm_loadu_si128(transmute(pixel));
            let result = _mm_mul_damp2_i8(batch, scale,mask,shift);
            _mm_storeu_si128(transmute(pixel),result);
        }
    }
}


pub fn print_i128(a:__m128i){
    unsafe{
        let c_ptr:*mut i32 = transmute(&a);
        for k in 0..4{ 
            print!("[{}]", *c_ptr.offset(k) );
        }
        print!("\n");
    }
}
pub fn print_i128_bytes(a:__m128i){
    unsafe{
        let c_ptr:*mut u8 = transmute(&a);
        for k in 0..16{ 
            print!("[{:4}]", *c_ptr.offset(k) );
        }
        print!("\n");
    }
}
