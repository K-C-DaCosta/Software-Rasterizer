use nalgebra::{Vector2, Vector4};
use std::fmt::Debug;
use std::ops::*;

use super::math_util::fixed_point;

pub struct Rect2D<T: 'static> {
    pub x: T,
    pub y: T,
    pub w: T,
    pub h: T,
}
impl<T: 'static + Sub> Rect2D<T> {
    pub fn new(x: T, y: T, w: T, h: T) -> Rect2D<T> {
        Rect2D { x, y, w, h }
    }
}

impl From<Rect2D<f32>> for Rect2D<i32> {
    fn from(a: Rect2D<f32>) -> Self {
        Rect2D {
            x: a.x as i32,
            y: a.y as i32,
            w: a.w as i32,
            h: a.h as i32,
        }
    }
}

pub struct AABB2D<T: Debug + PartialEq + Copy + 'static> {
    pub tl: Vector2<T>,
    pub br: Vector2<T>,
}

impl<T> AABB2D<T>
where
    T: 'static + Default + Sub<T, Output = T> + PartialOrd + PartialEq + Copy + Debug,
{
    pub fn new() -> AABB2D<T> {
        AABB2D {
            tl: Vector2::new(T::default(), T::default()),
            br: Vector2::new(T::default(), T::default()),
        }
    }
    pub fn to_rect(&self) -> Rect2D<T> {
        Rect2D {
            x: self.tl.x,
            y: self.tl.y,
            w: self.br.x - self.tl.x,
            h: self.br.y - self.tl.y,
        }
    }
}

pub trait AABBoxOps<T: Debug + PartialEq + Copy + 'static + Sub> {
    type WOUT;
    type HOUT;
    fn update_aabb(&mut self, pos_list: &[Vector4<T>; 3]);
    fn clip_box(&mut self, min_x: T, max_x: T, min_y: T, max_y: T);
    fn width(&self) -> Self::WOUT;
    fn height(&self) -> Self::HOUT;
}

impl<T> AABBoxOps<T> for AABB2D<T>
where
    T: Sub
        + Default
        + Debug
        + PartialEq
        + Copy
        + PartialOrd
        + fixed_point::CompOps
        + 'static
        + std::fmt::Display,
{
    type WOUT = <T as Sub>::Output;
    type HOUT = <T as Sub>::Output;

    fn clip_box(&mut self, min_x: T, max_x: T, min_y: T, max_y: T) {
        self.tl.x = self.tl.x.pmax(min_x);
        self.tl.y = self.tl.y.pmax(min_y);
        self.br.x = self.br.x.pmin(max_x);
        self.br.y = self.br.y.pmin(max_y);
    }

    fn width(&self) -> Self::WOUT {
        self.br.x - self.tl.x
    }

    fn height(&self) -> Self::HOUT {
        self.br.y - self.tl.y
    }

    fn update_aabb(&mut self, pos_list: &[Vector4<T>; 3]) {
        self.tl = pos_list[0].xy();
        self.br = pos_list[0].xy();
        for pos in pos_list {
            // println!("pos = {} tl = {}, br = {}",pos, self.tl,self.br);
            self.tl.x = self.tl.x.pmin(pos.x);
            self.tl.y = self.tl.y.pmin(pos.y);
            self.br.x = self.br.x.pmax(pos.x);
            self.br.y = self.br.y.pmax(pos.y);
            // println!("pos = {} tl = {}, br = {}",pos, self.tl,self.br);
        }
    }
}



