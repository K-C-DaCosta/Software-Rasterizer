use super::super::graphics::TriangleData;
use super::fixed_point::*;
use super::MatrixStack;
use nalgebra::{Vector2, Vector4};
use num_traits::identities;
use simba::scalar;
use std::cmp;
use std::fmt;
use std::ops;

pub struct BaryMatrixCoefs<T: Copy + PartialEq + fmt::Debug + 'static> {
    p0: Vector4<T>,
    v1: Vector4<T>,
    v2: Vector4<T>,
    a: T,
    b: T,
    d: T,
    inv_denom: T,
}
pub struct BaryWeights<T: Copy + PartialEq + fmt::Debug + 'static> {
    pub w: Vector4<T>,
}

impl<T> ops::Index<usize> for BaryWeights<T>
where
    T: Copy + PartialEq + fmt::Debug + 'static,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.w[index]
    }
}

impl<T> BaryMatrixCoefs<T>
where
    T: Copy
        + 'static
        + Default
        + PartialEq
        + fmt::Debug
        + identities::Zero
        + identities::One
        + scalar::ClosedAdd
        + scalar::ClosedMul
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>
        + ops::Sub<T, Output = T>,
    Vector4<T>: ops::Sub<Vector4<T>, Output = Vector4<T>>,
{
    pub fn new(points: &[Vector4<T>; 3]) -> BaryMatrixCoefs<T> {
        let mut bmc = BaryMatrixCoefs {
            p0: Vector4::zeros(),
            v1: Vector4::zeros(),
            v2: Vector4::zeros(),
            a: T::zero(),
            b: T::zero(),
            d: T::zero(),
            inv_denom: T::default(),
        };
        bmc.update_weights(points);
        bmc
    }

    pub fn update_weights(&mut self, points: &[Vector4<T>; 3]) {
        self.p0 = points[0];
        self.v1 = points[1] - points[0];
        self.v2 = points[2] - points[0];
        self.a = self.v1.dot(&self.v1);
        self.b = self.v2.dot(&self.v1);
        self.d = self.v2.dot(&self.v2);
        self.inv_denom = T::one() / (self.b * self.b - self.a * self.d);
    }

    pub fn calc_bary_weights(&self, pos: Vector4<T>) -> BaryWeights<T> {
        let v0 = pos - self.p0;
        let e = v0.dot(&self.v1);
        let f = v0.dot(&self.v2);
        let t = (self.b * f - e * self.d) * self.inv_denom;
        let s = (e * self.b - self.a * f) * self.inv_denom;
        BaryWeights {
            w: Vector4::new(T::one() - t - s, t, s, T::zero()),
        }
    }
}

/// A simple struct where:
/// 
/// a := v0.y-v1.y\
/// b := v1.x-v0.x\
/// c := v0.x*v1.y-v0.y*v1.x
/// 
pub struct EdgeWeights<T> {
    pub a: T,
    pub b: T,
    pub c: T,
}

///Evaluates edge function (mostly used for weights and not the actual function)
#[inline]
pub fn edge_function<T>(v0: Vector4<T>, v1: Vector4<T>, p: Vector4<T>) -> (T, EdgeWeights<T>)
where
    T: Copy
        + 'static
        + fmt::Debug
        + PartialEq
        + ops::Mul<T, Output = T>
        + ops::Sub<T, Output = T>
        + ops::Add<T, Output = T>,
{
    let a = v0.y - v1.y;
    let b = v1.x - v0.x;
    let c = v0.x * v1.y - v0.y * v1.x;
    let f = a * p.x + b * p.y + c;
    (f, EdgeWeights { a, b, c })
}

pub fn segment_v_plane<'a, T>(
    plane_p0: &Vector4<T>,
    plane_norm: &Vector4<T>,
    s0: &Vector4<T>,
    s1: &Vector4<T>,
) -> (Vector4<T>, T)
where
    T: Copy
        + 'static
        + Default
        + fmt::Debug
        + PartialEq
        + scalar::ClosedAdd
        + scalar::ClosedMul
        + identities::Zero
        + ops::Sub<T, Output = T>
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>,
    Vector4<T>: ops::Sub<Vector4<T>, Output = Vector4<T>>,
{
    let num = *plane_p0 - *s0;
    let denom = *s1 - *s0;
    let t = num.dot(plane_norm) / denom.dot(plane_norm);
    (denom * t + s0, t)
}

pub struct Precomputed2DTriangleSDF<T: Copy + PartialEq + fmt::Debug + 'static> {
    e0: Vector2<T>,
    e1: Vector2<T>,
    e2: Vector2<T>,
    inv_e0: T,
    inv_e1: T,
    inv_e2: T,
    s: T,
    p: [Vector2<T>; 3],
}

impl<'a, T> Precomputed2DTriangleSDF<T>
where
    T: Copy
        + 'static
        + Sized
        + Default
        + fmt::Debug
        + PartialEq
        + PartialOrd
        + CompOps
        + scalar::ClosedAdd
        + scalar::ClosedMul
        + identities::Zero
        + identities::One
        + ops::Neg<Output = T>
        + ops::Sub<T, Output = T>
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>,
    Vector2<T>: ops::Sub<Vector2<T>, Output = Vector2<T>>,
{
    pub fn new(p: &[Vector4<T>; 3]) -> Precomputed2DTriangleSDF<T> {
        let (e0, e1, e2) = (
            p[1].xy() - p[0].xy(),
            p[2].xy() - p[1].xy(),
            p[0].xy() - p[2].xy(),
        );

        let inv_e0 = T::one() / e0.dot(&e0);
        let inv_e1 = T::one() / e1.dot(&e1);
        let inv_e2 = T::one() / e2.dot(&e2);

        let s = (e0.x * e2.y - e0.y * e2.x).psign();

        Precomputed2DTriangleSDF {
            e0,
            e1,
            e2,
            inv_e0,
            inv_e1,
            inv_e2,
            s,
            p: [p[0].xy(), p[1].xy(), p[2].xy()],
        }
    }
    pub fn squared_signed_distance(&self, pi: Vector2<T>) -> T {
        let p = &self.p;
        let (e0, e1, e2) = (self.e0, self.e1, self.e2);
        let (v0, v1, v2) = (pi - p[0], pi - p[1], pi - p[2]);

        let pq0 = v0 - e0 * (v0.dot(&e0) * self.inv_e0).pmin(T::one()).pmax(T::zero());
        let pq1 = v1 - e1 * (v1.dot(&e1) * self.inv_e1).pmin(T::one()).pmax(T::zero());
        let pq2 = v2 - e2 * (v2.dot(&e2) * self.inv_e2).pmin(T::one()).pmax(T::zero());
        let s = self.s;

        let d1 = Vector2::new(pq0.dot(&pq0), s * (v0.x * e0.y - v0.y * e0.x));
        let d2 = Vector2::new(pq1.dot(&pq1), s * (v1.x * e1.y - v1.y * e1.x));
        let d3 = Vector2::new(pq2.dot(&pq2), s * (v2.x * e2.y - v2.y * e2.x));

        let d = Vector2::new(d3.x.pmin(d2.x.pmin(d1.x)), d3.y.pmin(d2.y.pmin(d1.y)));
        -d.x * d.y.psign()
    }
}

//helper function: cell_has_tri(..) returns true if the cell intersects the triangle
pub fn cell_overlaps_triangle<T>(
    tx0: i32,
    ty0: i32,
    tw: i32,
    th: i32,
    triangle: &Precomputed2DTriangleSDF<T>,
) -> bool
where
    T: Copy
        + 'static
        + Default
        + fmt::Debug
        + PartialEq
        + PartialOrd
        + CompOps
        + scalar::ClosedAdd
        + scalar::ClosedMul
        + identities::Zero
        + identities::One
        + ScalarConvert<i32>
        + ops::Neg<Output = T>
        + ops::Sub<T, Output = T>
        + ops::Add<T, Output = T>
        + ops::Mul<T, Output = T>
        + ops::Div<T, Output = T>,
    Vector2<T>: ops::Sub<Vector2<T>, Output = Vector2<T>>,
{
    let center = Vector2::new(T::convert(tx0 + tw / 2), T::convert(ty0 + th / 2));
    let tile_rad = T::convert(tw * tw / 4 + th * th / 4);
    let dist_squared = triangle.squared_signed_distance(center);
    dist_squared < tile_rad
}

pub fn attribs_to_screen_space(
    attribs: &mut [TriangleData; 3],
    attrib_mask: &mut [bool; 3],
    stack: &MatrixStack,
) {
    let viewport = &stack.viewport;
    let projection = &stack.projection;
    let mut eye_depths = [0.0; 3];
    const NUM_ATTRIBS: usize = 2;

    for (attr_index, attr) in attribs.iter_mut().enumerate() {
        {
            let verts = attr.get_mut(0);

            if attrib_mask[attr_index] == false {
                continue;
            }

            //back-face culling code here
            let d1 = (verts[0] - verts[1]).xyz();
            let d2 = (verts[1] - verts[2]).xyz();
            let normal = d1.cross(&d2);
            if verts[0].xyz().dot(&normal) >= 0.0 {
                attrib_mask[attr_index] = false; //flag as "don't draw"
                continue;
            }

            for k in 0..verts.len() {
                let v = verts[k];
                let v_depth = v.z;

                //compute clip coordinate
                let clip_coords = projection * v;

                //convert clip coordinates to ndc
                let ndc = clip_coords * (1.0 / clip_coords.w);

                //convert from ndc to screenspace
                let mut screen_space = viewport * ndc;
                screen_space.z = 0.0;
                screen_space.w = 1.0;

                //write scren space coordinate
                verts[k] = screen_space;
                eye_depths[k] = v_depth;
            }
        }

        //fixup triangle winding order by flipping attributes
        let is_clockwise = {
            let verts = attr.get_mut(0);
            let (area, _) = edge_function(verts[0], verts[1], verts[2]);
            area < 0.0
        };

        if is_clockwise && attrib_mask[attr_index] {
            // attrib_mask[attr_index] = false;
            //swap vector attributes(by mutating attr struct)
            for k in 0..NUM_ATTRIBS + 1 {
                let generic_attrib = attr.get_mut(k);
                let temp = generic_attrib[0];
                generic_attrib[0] = generic_attrib[1];
                generic_attrib[1] = temp;
            }

            //swap eye depths (not yet written to attr struct)
            let temp = eye_depths[0];
            eye_depths[0] = eye_depths[1];
            eye_depths[1] = temp;
        }

        //write out eye_depths to the scalar atributes
        *attr.get_scalar_mut(0) = eye_depths;
    }
}

/// Clips triangle in eye space.
/// # Arguments
/// * `attribs` - A fixed array of records that contain triangle information (including attributes like normals and uv's)
///
/// Returns the number of verticies behind the plane
pub fn clip_triangle(attribs: &mut [TriangleData; 3], near_z: f32) -> usize {
    //currently only clips two attributes( normal and uv )
    const MAX_ATTRIBS: usize = 2;

    let p0 = Vector4::new(0., 0., near_z, 1.0);
    let normal = Vector4::new(0., 0., -1., 0.);
    let mut outside_count = 0;
    let mut first_outside_index = !0;
    let mut first_inside_index = !0;
    let primary = attribs[0];

    let lerp = |a, b, t| (b - a) * t + a;

    for (k, p) in primary.get(0).iter().enumerate() {
        let disp = p - p0;
        let is_outside = disp.dot(&normal) < 0.0;
        if is_outside {
            if first_outside_index == !0 {
                first_outside_index = k;
            }
            outside_count += 1;
        } else {
            if first_inside_index == !0 {
                first_inside_index = k;
            }
        }
    }

    if outside_count == 1 {
        //one point is behind the plane so we split into two triangles
        let right = (first_outside_index + 1) % 3;
        let left = (first_outside_index + 2) % 3;

        let ((right_intersection, tr), (left_intersection, tl)) = {
            let points = attribs[0].get(0);
            (
                segment_v_plane(&p0, &normal, &points[first_outside_index], &points[right]),
                segment_v_plane(&p0, &normal, &points[first_outside_index], &points[left]),
            )
        };

        //calculate triangle 1
        let triangle1_verts = attribs[1].get_mut(0);
        triangle1_verts[0] = left_intersection;
        triangle1_verts[1] = primary.get(0)[right];
        triangle1_verts[2] = primary.get(0)[left];

        //compute clipped attributes(normal and uv are attribs 1 and 2 respectively)
        for k in 1..=MAX_ATTRIBS {
            let triangle1_attrib = attribs[1].get_mut(k);
            let primary_attrib = primary.get(k);
            triangle1_attrib[0] = lerp(
                primary_attrib[first_outside_index],
                primary_attrib[left],
                tl,
            );
            triangle1_attrib[1] = primary_attrib[right];
            triangle1_attrib[2] = primary_attrib[left];
        }

        //calculate triangle 2
        let triangle2_verts = attribs[2].get_mut(0);
        triangle2_verts[0] = left_intersection;
        triangle2_verts[1] = right_intersection;
        triangle2_verts[2] = primary.get(0)[right];

        //compute clipped attributes(normal and uv are attribs 1 and 2 respectively)
        for k in 1..=MAX_ATTRIBS {
            let triangle2_attrib = attribs[2].get_mut(k);
            let primary_attrib = primary.get(k);
            triangle2_attrib[0] = lerp(
                primary_attrib[first_outside_index],
                primary_attrib[left],
                tl,
            );
            triangle2_attrib[1] = lerp(
                primary_attrib[first_outside_index],
                primary_attrib[right],
                tr,
            );
            triangle2_attrib[2] = primary_attrib[right];
        }
    } else if outside_count == 2 {
        //two points are behind the plane so we only generate one triangle

        let left = (first_inside_index + 1) % 3;
        let right = (first_inside_index + 2) % 3;
        let first_point = attribs[0].get(0)[first_inside_index];

        let ((right_intersection, tr), (left_intersection, tl)) = {
            let points = attribs[0].get(0);
            (
                segment_v_plane(&p0, &normal, &points[first_inside_index], &points[right]),
                segment_v_plane(&p0, &normal, &points[first_inside_index], &points[left]),
            )
        };

        //initilize verts
        let triangle1_verts = attribs[1].get_mut(0);
        triangle1_verts[0] = first_point;
        triangle1_verts[1] = left_intersection;
        triangle1_verts[2] = right_intersection;

        //compute clipped attributes(normal and uv are attribs 1 and 2 respectively)
        for k in 1..=MAX_ATTRIBS {
            let triangle1_attrib = attribs[1].get_mut(k);
            let primary_attrib = primary.get(k);
            triangle1_attrib[0] = primary_attrib[first_inside_index];
            triangle1_attrib[1] =
                lerp(primary_attrib[first_inside_index], primary_attrib[left], tl);
            triangle1_attrib[2] = lerp(
                primary_attrib[first_inside_index],
                primary_attrib[right],
                tr,
            );
        }
    }
    outside_count
}
