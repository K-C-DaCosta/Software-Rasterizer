// use super::fixed_point::*;

// use std::fmt;
// use std::ops;

// #[derive(Copy, Clone)]
// pub struct FixedVec4<T> {
//     pub data: [T; 4],
// }

// pub trait VecOps<T> {
//     fn dot(&self, rhs: &Self) -> T;
// }

// impl<T> FixedVec4<T>
// where
//     T: Default + Copy,
// {
//     pub fn new(x: T, y: T, z: T, w: T) -> FixedVec4<T> {
//         FixedVec4 { data: [x, y, z, w] }
//     }

//     pub fn zeros() -> FixedVec4<T> {
//         FixedVec4 {
//             data: [T::default(); 4],
//         }
//     }
//     pub fn x(&self) -> T {
//         self.data[0]
//     }
//     pub fn y(&self) -> T {
//         self.data[1]
//     }
//     pub fn z(&self) -> T {
//         self.data[2]
//     }
//     pub fn w(&self) -> T {
//         self.data[3]
//     }
// }

// impl<T> VecOps<T> for FixedVec4<T>
// where
//     T: ops::Mul<T, Output = T>
//         + Copy
//         + Default
//         + ops::AddAssign<T>,
//     Self: ops::Index<usize, Output = T>,
// {
//     fn dot(&self, rhs: &Self) -> T {
//         let mut result = T::default();
//         for k in 0..4 {
//             result += self[k] * rhs[k];
//         }
//         result
//     }
// }

// impl<T> ops::Add<Self> for FixedVec4<T>
// where
//     T: ops::Add<T, Output = T> + Default + Copy,
// {
//     type Output = Self;

//     fn add(self, other: Self) -> Self {
//         FixedVec4 {
//             data: [
//                 self.x() + other.x(),
//                 self.y() + other.y(),
//                 self.z() + other.z(),
//                 self.w() + other.w(),
//             ],
//         }
//     }
// }

// impl<T> ops::Sub<Self> for FixedVec4<T>
// where
//     T: ops::Sub<T, Output = T> + Default + Copy,
// {
//     type Output = Self;

//     fn sub(self, other: Self) -> Self {
//         FixedVec4 {
//             data: [
//                 self.x() - other.x(),
//                 self.y() - other.y(),
//                 self.z() - other.z(),
//                 self.w() - other.w(),
//             ],
//         }
//     }
// }

// impl<T> ops::Mul<i32> for FixedVec4<T>
// where
//     T: ops::Mul<T, Output = T> + Default + Copy + ScalarConvert<i32>,
// {
//     type Output = Self;
//     fn mul(self, other: i32) -> Self {
//         let scalar: T = T::convert(other);
//         FixedVec4 {
//             data: [
//                 self.x() * scalar,
//                 self.y() * scalar,
//                 self.z() * scalar,
//                 self.w() * scalar,
//             ],
//         }
//     }
// }

// impl<T> ops::Index<usize> for FixedVec4<T>
// where
//     T: Default + Copy,
// {
//     type Output = T;
//     fn index(&self, index: usize) -> &Self::Output {
//         &self.data[index]
//     }
// }

// impl<T> fmt::Display for FixedVec4<T>
// where
//     T: fmt::Display + Default + Copy,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "[{},{},{},{}]", self.x(), self.y(), self.z(), self.w())
//     }
// }
