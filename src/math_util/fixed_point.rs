use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::ops;

use num_traits::identities;
use typenum::*;

#[allow(non_camel_case_types)]
pub type fpi15_16 = BinaryFixed<i32, U16, U8, U8, U0, U5, U11, U0>;

#[allow(non_camel_case_types)]
pub type fpi24_7 = BinaryFixed<i32, U7, U4, U3, U0, U1, U3, U0>;

#[allow(non_camel_case_types)]
pub type fpi3_28 = BinaryFixed<i32, U28, U14, U14, U0, U10, U14, U0>;

#[allow(non_camel_case_types)]
pub type fpi11_20 = BinaryFixed<i32, U20, U10, U10, U0, U5, U10, U0>;

#[allow(non_camel_case_types)]
pub type fpi7_8 = BinaryFixed<i16, U8, U4, U4, U0, U3, U5, U0>;

///Fixed point math with binary scaling
#[derive(Copy, Clone, Default, Debug)]
pub struct BinaryFixed<T: Sized + Copy, F, ML, MR, MC, DL, DR, DC> {
    pub data: T,
    _dummy: PhantomData<(F, ML, MR, MC, DL, DR, DC)>,
}

impl<T, F, ML, MR, MC, DL, DR, DC> BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    T: Default + Copy,
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
{
    #[inline]
    pub fn new(data: T) -> BinaryFixed<T, F, ML, MR, MC, DL, DR, DC> {
        BinaryFixed {
            data,
            _dummy: PhantomData::default(),
        }
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ops::Add<Self> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: ops::Add<T, Output = T> + Copy + Default,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        BinaryFixed::new(self.data + rhs.data)
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ops::AddAssign<Self>
    for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: ops::AddAssign<T> + Copy + Default,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.data += rhs.data
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ops::Sub<Self> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: ops::Sub<T, Output = T> + Copy + Default,
{
    type Output = BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        BinaryFixed::new(self.data - rhs.data)
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ops::SubAssign<Self>
    for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: ops::SubAssign<T> + Copy + Default,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.data -= rhs.data
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ops::Mul<Self> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned + Default,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: ops::Mul<T, Output = T> + ops::Shr<T, Output = T> + Copy + Default + ScalarConvert<i32>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let coef_a = self.data >> T::convert(ML::I32);
        let coef_b = rhs.data >> T::convert(MR::I32);
        let product = coef_a * coef_b >> T::convert(MC::I32);
        BinaryFixed::new(product)
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ops::Mul<&Self> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned + Default,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: ops::Mul<T, Output = T> + ops::Shr<T, Output = T> + Copy + Default + ScalarConvert<i32>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        let coef_a = self.data >> T::convert(ML::I32);
        let coef_b = rhs.data >> T::convert(MR::I32);
        let product = coef_a * coef_b >> T::convert(MC::I32);
        BinaryFixed::new(product)
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ops::MulAssign<Self>
    for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: ops::Mul<T, Output = T> + ops::MulAssign<T> + Copy + Default,
    Self: Copy + ops::Mul<Self, Output = Self>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = (*self) * rhs;
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ops::Div<Self> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned + Default,
    ML: Unsigned,
    MR: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    MC: Unsigned,
    DC: Unsigned,
    T: ops::Div<T, Output = T> + ops::Shl<T, Output = T> + Copy + Default + ScalarConvert<i32>,
{
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let numer = self.data << T::convert(DL::I32);
        let denom = rhs.data;
        let quotient = (numer / denom) << T::convert(DR::I32);
        BinaryFixed::new(quotient)
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ops::DivAssign<Self>
    for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: ops::DivAssign<T> + Copy + Default,
    Self: Copy + ops::Div<Self, Output = Self>,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = (*self) / rhs;
    }
}

//unary negation
impl<T, F, ML, MR, MC, DL, DR, DC> ops::Neg for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: Default + Copy + ops::Neg<Output = T>,
{
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        BinaryFixed::new(-self.data)
    }
}

///Binary and
impl<T, F, ML, MR, MC, DL, DR, DC> ops::BitAnd<T> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    T: Default + Copy + ops::BitAnd<T, Output = T>,
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: T) -> Self::Output {
        BinaryFixed::new(self.data & rhs)
    }
}
///Binary or
impl<T, F, ML, MR, MC, DL, DR, DC> ops::BitOr<T> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: Default + Copy + ops::BitOr<T, Output = T>,
{
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: T) -> Self::Output {
        BinaryFixed::new(self.data | rhs)
    }
}

///Binary shift left
impl<T, F, ML, MR, MC, DL, DR, DC> ops::Shl<T> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: Default + Copy + ops::Shl<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn shl(self, rhs: T) -> Self::Output {
        let shl = self.data << rhs;
        BinaryFixed::new(shl)
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> cmp::PartialEq for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: Default + Copy + cmp::PartialEq<T>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data.eq(&other.data)
    }
}

///Binary shift right
impl<T, F, ML, MR, MC, DL, DR, DC> ops::Shr<T> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: Default + Copy + ops::Shr<T, Output = T>,
{
    type Output = Self;

    #[inline]
    fn shr(self, rhs: T) -> Self::Output {
        let shr = self.data >> rhs;
        BinaryFixed::new(shr)
    }
}

impl<F, ML, MR, MC, DL, DR, DC> fmt::Display for BinaryFixed<i32, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let converted_number = self.data as f32 / (1 << F::I32) as f32;
        write!(f, "{:.10}", converted_number)
    }
}

impl<F, ML, MR, MC, DL, DR, DC> fmt::Display for BinaryFixed<i16, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let converted_number = self.data as f32 / (1 << F::I32) as f32;
        write!(f, "{:.10}", converted_number)
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> identities::Zero for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
    T: ScalarConvert<i32> + Copy + ops::Add<T, Output = T> + cmp::PartialEq + Default,
{
    fn zero() -> Self {
        BinaryFixed::new(T::convert(0))
    }

    fn is_zero(&self) -> bool {
        self.data == T::convert(0)
    }
}

pub trait ScalarConvert<T> {
    fn convert(a: T) -> Self;
}

impl ScalarConvert<u32> for u8 {
    #[inline]
    fn convert(a: u32) -> Self {
        a as Self
    }
}

impl ScalarConvert<i8> for i16 {
    #[inline]
    fn convert(a: i8) -> Self {
        a as Self
    }
}

impl ScalarConvert<f32> for i16 {
    #[inline]
    fn convert(a: f32) -> Self {
        a as Self
    }
}
impl ScalarConvert<i16> for i16 {
    #[inline]
    fn convert(a: i16) -> Self {
        a
    }
}

impl ScalarConvert<i32> for i16 {
    #[inline]
    fn convert(a: i32) -> Self {
        a as Self
    }
}

impl ScalarConvert<u32> for i16 {
    #[inline]
    fn convert(a: u32) -> Self {
        a as Self
    }
}

impl ScalarConvert<i8> for i32 {
    #[inline]
    fn convert(a: i8) -> Self {
        a as Self
    }
}

impl ScalarConvert<i16> for i32 {
    #[inline]
    fn convert(a: i16) -> Self {
        a as Self
    }
}

impl ScalarConvert<i32> for i32 {
    #[inline]
    fn convert(a: i32) -> Self {
        a
    }
}

impl ScalarConvert<f32> for i32 {
    #[inline]
    fn convert(a: f32) -> Self {
        a as i32
    }
}

impl ScalarConvert<i8> for u32 {
    #[inline]
    fn convert(a: i8) -> Self {
        a as Self
    }
}

impl ScalarConvert<i32> for f32 {
    #[inline]
    fn convert(a: i32) -> Self {
        a as Self
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ScalarConvert<f32> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    T: Copy + Default + ScalarConvert<f32>,
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
{
    #[inline]
    fn convert(a: f32) -> Self {
        let conversion = (1 << F::I32) as f32 * a;
        BinaryFixed::new(T::convert(conversion))
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ScalarConvert<u32> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    T: ScalarConvert<i32> + Copy + Default,
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
{
    #[inline]
    fn convert(a: u32) -> Self {
        let conversion = (a << F::I32) as i32;
        let b = T::convert(conversion);
        BinaryFixed::new(b)
    }
}

impl<T, F, ML, MR, MC, DL, DR, DC> ScalarConvert<Self> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    T: ScalarConvert<i32> + Copy + Default,
    F: Unsigned,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
{
    #[inline]
    fn convert(a: Self) -> Self {
        a
    }
}

pub trait FixedMathUtils<T> {
    ///returns tUnsigned part of the fractional number (in fixed point format)
    fn floor(self) -> Self;
    ///returns the fractional part of the fixed point number (in fixed point format)
    fn fract(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn smooth_step(self) -> Self;
    ///returns the integer part of the fixed point number(not in fixed point format)
    ///(if T=int32 then it does what we usually want)
    fn int_part(self) -> T;
}

impl<T, F, ML, MR, MC, DL, DR, DC> FixedMathUtils<T> for BinaryFixed<T, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned + Default + Copy,
    ML: Unsigned + Copy,
    MR: Unsigned + Copy,
    MC: Unsigned + Copy,
    DL: Unsigned + Copy,
    DR: Unsigned + Copy,
    DC: Unsigned + Copy,
    T: Copy
        + Default
        + ops::BitAnd<T, Output = T>
        + ops::BitOr<T, Output = T>
        + ops::Shr<T, Output = T>
        + ops::Shl<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Sub<T, Output = T>
        + ops::Add<T, Output = T>
        + ops::Div<T, Output = T>
        + ScalarConvert<i32>,
    Self: ops::Mul<Self, Output = Self>,
{
    #[inline]
    fn floor(self) -> Self {
        let mask = (1 << F::I32) - 1;
        self & T::convert(!mask)
    }
    #[inline]
    fn fract(self) -> Self {
        let mask = (1 << F::I32) - 1;
        self & T::convert(mask)
    }
    fn sqrt(self) -> Self {
        panic!("sqrt(...) Not implemented")
    }
    fn sin(self) -> Self {
        panic!("sin(...) Not implemented")
    }
    fn cos(self) -> Self {
        panic!("cos(...) Not implemented")
    }
    #[inline]
    fn smooth_step(self) -> Self {
        let x = self;
        let _3 = Self::convert(3);
        let _2 = Self::convert(2);
        x * x * (_3 - _2 * x)
    }
    #[inline]
    fn int_part(self) -> T {
        (self >> T::convert(F::I32)).data
    }
}

///A Wrapper trait for primitive types 
/// which expose generic min,max and sign 
///functions
pub trait CompOps {
    fn pmin(self, b: Self) -> Self;
    fn pmax(self, b: Self) -> Self;
    fn psign(&self) -> Self;
}

impl CompOps for f32 {
    fn pmin(self, b: Self) -> Self {
        self.min(b)
    }
    fn pmax(self, b: Self) -> Self {
        self.max(b)
    }
    fn psign(&self) -> Self {
        self.signum()
    }
}

impl CompOps for i32 {
    fn pmin(self, b: Self) -> Self {
        self.min(b)
    }
    fn pmax(self, b: Self) -> Self {
        self.max(b)
    }
    fn psign(&self) -> Self {
        self.signum()
    }
}

impl<F, ML, MR, MC, DL, DR, DC> CompOps for BinaryFixed<i32, F, ML, MR, MC, DL, DR, DC>
where
    F: Unsigned + Default,
    ML: Unsigned,
    MR: Unsigned,
    MC: Unsigned,
    DL: Unsigned,
    DR: Unsigned,
    DC: Unsigned,
{
    fn pmin(self, b: Self) -> Self {
        BinaryFixed::new(self.data.min(b.data))
    }
    fn pmax(self, b: Self) -> Self {
        BinaryFixed::new(self.data.max(b.data))
    }
    fn psign(&self) -> Self {
        BinaryFixed::new(self.data.signum() << F::I32)
    }
}
