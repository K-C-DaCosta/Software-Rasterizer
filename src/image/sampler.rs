#![allow(unused_variables)]

use super::{Buffer2D};
use super::super::math_util::fixed_point::*; 
use std::fmt;
use nalgebra::{Vector4};

const WT_W:usize = 128;
const WT_H:usize = 128;

lazy_static! {
    static ref WEIGHT_TABLE_FLOAT:[f32;4*WT_W*WT_H] = {
        let mut table = [0f32;4*WT_W*WT_H];
        for y in 0..WT_H as i32{
            for x in 0..WT_W as i32{
                
                let index = (WT_W as i32 *y  + x) as usize * 4; 
                let u = (x as f32+0.5)/ WT_W as f32 ; 
                let v = (y as f32+0.5)/ WT_H as f32;  

                let one = 1.0;
                let wtl = (one-u)*(one-v);
                let wtr = (one-v)*u;
                let wbl = (one-u)*v;  
                let wbr = u*v; 

                table[index+0] = wtl;
                table[index+1] = wtr;
                table[index+2] = wbl;
                table[index+3] = wbr;
            }
        }
        table
    };
}


pub trait Sampler2D<CT,BT:Copy + fmt::Display + Default >{
    fn interpolate<T:Buffer2D<BT>>(&self, x:CT,y:CT,buffer:& T)->BT;
    fn uv_map<T:Buffer2D<BT>>(&self,uv:Vector4<f32>,buffer:& T)->BT;
}

pub struct NearestSampler2D;


pub struct LinearSampler2D; 


impl Sampler2D<f32,u32> for NearestSampler2D{
    fn interpolate<T:Buffer2D<u32>>(&self, x: f32, y: f32, buffer: & T)->u32{
         buffer.get((y as u32,x as u32)) 
    }
    fn uv_map<T:Buffer2D<u32>>(&self,uv:Vector4<f32>,buffer:& T)->u32{
        let w = buffer.width() as f32; 
        let h = buffer.height() as f32; 
        let u = (w*uv.x).min(w-1.0).max(0.0);
        let v = (h*(1.0-uv.y)).min(h-1.0).max(0.0);
        self.interpolate(u, v, buffer)
    }
}

impl Sampler2D<f32,u32> for LinearSampler2D{
    fn interpolate<T:Buffer2D<u32>>(&self, x: f32, y: f32, src_buffer: & T)->u32{
        let u =  x - (x as i32) as f32;
        let v =  y - (y as i32) as f32;
        let x0 = x as u32;
        let y0 = y as u32; 
        let w = src_buffer.width();
        let h = src_buffer.height();

        //clamping 'adjacent indices'
        let x0_1 = (x0+1).min(w-1);
        let y0_1 = (y0+1).min(h-1);

        //grab adjacent colors in the source image 
        let mut tl = src_buffer.get((y0  ,x0  ));
        let mut tr = src_buffer.get((y0  ,x0_1));
        let mut bl = src_buffer.get((y0_1,x0  ));
        let mut br = src_buffer.get((y0_1,x0_1));
        
        //compute linear interpolatqion weights 
        let one = 1.0;
        let wtl = (one-u)*(one-v);
        let wtr = (one-v)*u;
        let wbl = (one-u)*v;  
        let wbr = u*v; 

        let bi_lerp = |tl,tr,bl,br,wtl,wtr,wbl,wbr| tl as f32 *wtl +tr as f32 * wtr+bl as f32 * wbl + br as f32 * wbr ;
        
        
        //sequentially calculate interpolated color channels 
        let blue = bi_lerp(tl&0xff,tr&0xff,bl&0xff,br &0xff,wtl,wtr,wbl,wbr) as u32;
        
        tl>>=8;
        tr>>=8;
        bl>>=8;
        br>>=8; 
        
        let green = bi_lerp(tl&0xff,tr&0xff,bl&0xff,br &0xff,wtl,wtr,wbl,wbr) as u32 ;
        
        tl>>=8;
        tr>>=8;
        bl>>=8;
        br>>=8; 
        
        let red = bi_lerp(tl&0xff,tr&0xff,bl&0xff,br &0xff,wtl,wtr,wbl,wbr) as u32;
        
        tl>>=8;
        tr>>=8;
        bl>>=8;
        br>>=8;
        
        let alpha = bi_lerp(tl&0xff,tr&0xff,bl&0xff,br &0xff,wtl,wtr,wbl,wbr) as u32;
        
        //return interpolated color 
        ((alpha&0xff) << 24) | ((red&0xff) << 16) | ((green&0xff) << 8) | (blue&0xff) 
    }
    fn uv_map<T:Buffer2D<u32>>(&self,uv:Vector4<f32>,buffer:& T)->u32{
        let w = buffer.width() as f32; 
        let h = buffer.height() as f32; 

        let u = (w*uv.x).min(w-1.0).max(0.0);
        let v = (h*(1.0-uv.y)).min(h-1.0).max(0.0);

        self.interpolate(u, v, buffer)
    }
}

// impl <V:FixedMathUtils<i32>> Sampler2D<V,u32> for NearestSampler2D{
//     fn interpolate<T:Buffer2D<u32>>(&self, x:V, y:V, buffer: & T)->u32{
//          buffer.get((y.floor().int_part() as u32,x.floor().int_part() as u32)) 
//     }

//     fn uv_map<T:Buffer2D<u32>>(&self,uv:Vector4<f32>,buffer:& T)->u32{
//         panic!("not implemented")
//     }
// }

// impl Sampler2D<fpi15_16,u32> for LinearSampler2D
// where 
// {
//     fn interpolate<T:Buffer2D<u32>>(&self,x:fpi15_16,y:fpi15_16,src_buffer:& T)->u32{
//         let u = x.fract();
//         let v = y.fract();
//         let x0 = x.floor().int_part() as u32;
//         let y0 = y.floor().int_part() as u32; 
//         let w = src_buffer.width() ;
//         let h = src_buffer.height();

//         //clamping 'adjacent indices'
//         let x0_1 = (x0+1).min(w-1);
//         let y0_1 = (y0+1).min(h-1);

//         //grab adjacent colors in the source image 
//         let mut tl = src_buffer.get((y0  ,x0  ));
//         let mut tr = src_buffer.get((y0  ,x0_1));
//         let mut bl = src_buffer.get((y0_1,x0  ));
//         let mut br = src_buffer.get((y0_1,x0_1));

//         //compute bilinear interpolation weights (with fixed point maths)
//         let one = fpi15_16::convert(1);
//         let wtl = (one-u)*(one-v);
//         let wtr = (one-v)*u;
//         let wbl = (one-u)*v;  
//         let wbr = u*v; 

//         //compute colors sequentially
//         let blue = fpi15_16::convert(tl&0xff)*wtl + fpi15_16::convert(tr&0xff)*wtr +
//                    fpi15_16::convert(bl&0xff)*wbl + fpi15_16::convert(br&0xff)*wbr ;
//         tl>>=8;
//         tr>>=8;
//         bl>>=8;
//         br>>=8; 

//         let green = fpi15_16::convert(tl&0xff)*wtl + fpi15_16::convert(tr&0xff)*wtr +
//                     fpi15_16::convert(bl&0xff)*wbl + fpi15_16::convert(br&0xff)*wbr ;
//         tl>>=8;
//         tr>>=8;
//         bl>>=8;
//         br>>=8; 


//         let red = fpi15_16::convert(tl&0xff)*wtl + fpi15_16::convert(tr&0xff)*wtr +
//                   fpi15_16::convert(bl&0xff)*wbl + fpi15_16::convert(br&0xff)*wbr ;
//         tl>>=8;
//         tr>>=8;
//         bl>>=8;
//         br>>=8; 

//         let alpha = fpi15_16::convert(tl&0xff)*wtl + fpi15_16::convert(tr&0xff)*wtr +
//                     fpi15_16::convert(bl&0xff)*wbl + fpi15_16::convert(br&0xff)*wbr ;

//         let pix =(alpha.int_part() &0xff) << 24 | 
//                  (  red.int_part() &0xff) << 16 |
//                  (green.int_part() &0xff) << 8  |
//                  ( blue.int_part() &0xff);
//         pix as u32
//     }
//     fn uv_map<T:Buffer2D<u32>>(&self,uv:Vector4<f32>,buffer:& T)->u32{
//         panic!("not implemented")
//     }
// }

// impl Sampler2D<fpi7_8,u32> for LinearSampler2D
// where 
// {
//     fn interpolate<T:Buffer2D<u32>>(&self,x:fpi7_8,y:fpi7_8,src_buffer:& T)->u32{
//         let u = x.fract();
//         let v = y.fract();
//         let w = src_buffer.width() ;
//         let h = src_buffer.height();
//         let x0 = (x.floor().int_part() as u32).min(w);
//         let y0 = (y.floor().int_part() as u32).min(h); 


//         //clamping 'adjacent indices'
//         let x0_1 = (x0+1).min(w-1);
//         let y0_1 = (y0+1).min(h-1);

//         //grab adjacent colors in the source image 
//         let mut tl = src_buffer.get((y0  ,x0  ));
//         let mut tr = src_buffer.get((y0  ,x0_1));
//         let mut bl = src_buffer.get((y0_1,x0  ));
//         let mut br = src_buffer.get((y0_1,x0_1));

//         //compute bilinear interpolation weights (with fixed point maths)
//         let one = fpi7_8::convert(1);
//         let wtl = (one-u)*(one-v);
//         let wtr = (one-v)*u;
//         let wbl = (one-u)*v;  
//         let wbr = u*v; 

//         //compute colors sequentially
//         let blue = fpi7_8::convert(tl&0xff)*wtl + fpi7_8::convert(tr&0xff)*wtr +
//                    fpi7_8::convert(bl&0xff)*wbl + fpi7_8::convert(br&0xff)*wbr ;
//         tl>>=8;
//         tr>>=8;
//         bl>>=8;
//         br>>=8; 

//         let green = fpi7_8::convert(tl&0xff)*wtl + fpi7_8::convert(tr&0xff)*wtr +
//                     fpi7_8::convert(bl&0xff)*wbl + fpi7_8::convert(br&0xff)*wbr ;
//         tl>>=8;
//         tr>>=8;
//         bl>>=8;
//         br>>=8; 


//         let red = fpi7_8::convert(tl&0xff)*wtl + fpi7_8::convert(tr&0xff)*wtr +
//                   fpi7_8::convert(bl&0xff)*wbl + fpi7_8::convert(br&0xff)*wbr ;
//         tl>>=8;
//         tr>>=8;
//         bl>>=8;
//         br>>=8; 

//         let alpha = fpi7_8::convert(tl&0xff)*wtl + fpi7_8::convert(tr&0xff)*wtr +
//                     fpi7_8::convert(bl&0xff)*wbl + fpi7_8::convert(br&0xff)*wbr ;
       

//         let pix =(alpha.int_part() as u32 &0xff) << 24 | 
//                  (  red.int_part() as u32 &0xff) << 16 |
//                  (green.int_part() as u32 &0xff) << 8  |
//                  ( blue.int_part() as u32 &0xff);
//         pix as u32
//     }
//     fn uv_map<T:Buffer2D<u32>>(&self,uv:Vector4<f32>,buffer:& T)->u32{
//         panic!("not implemented")
//     }
// }

#[inline]
pub fn cached_bilinear_f32(u:f32, v:f32,data:&[f32;4])->f32{
    let x = ((WT_W as f32 * u) as i32).max(0).min(WT_W as i32 -1); 
    let y = ((WT_H as f32 * v) as i32).max(0).min(WT_H as i32 -1); 
    let index = (4*(y*WT_W as i32 + x)) as isize; 
    let ptr = unsafe{WEIGHT_TABLE_FLOAT.as_ptr().offset(index)};
    let mut approx = 0.0; 
    approx+=data[0]*unsafe{*ptr.offset(0)};
    approx+=data[1]*unsafe{*ptr.offset(1)};
    approx+=data[2]*unsafe{*ptr.offset(2)};
    approx+=data[3]*unsafe{*ptr.offset(3)};
    approx
}