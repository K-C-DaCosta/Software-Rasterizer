use super::math_util::geometry;
use super::{
    graphics::*,
    image::{
        buffer::{RowBuffer2D, TiledBuffer2D},
        Buffer2D,
    },
    math_util::MatrixStack,
};

use nalgebra::{Matrix4, Unit, Vector2, Vector3, Vector4};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, Read};

pub struct StaticModel {
    positions: Vec<Vector4<f32>>,
    normals: Vec<Vector4<f32>>,
    uvs: Vec<Vector4<f32>>,
    index_buffer: Vec<u32>,
}

impl StaticModel {
    pub fn new() -> StaticModel {
        StaticModel {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            index_buffer: Vec::new(),
        }
    }
    ///Generates a Cube for debuggin purposes
    pub fn gen_debug_cube(&mut self, delta: f32, delta_z: f32) {
        type V = Vector4<f32>;

        const PI: f32 = 3.14159;
        const PI_2: f32 = PI * 0.5;
        let front_plane = [
            V::new(delta, -delta, delta, 1.0),
            V::new(delta, delta, delta, 1.0),
            V::new(-delta, delta, delta, 1.0),
            V::new(-delta, -delta, delta, 1.0),
        ];
        let fp_norm = V::new(0., 0., 1.0, 0.0);
        //front[0-3]
        let rot_mat =
            Matrix4::<f32>::from_axis_angle(&Unit::new_unchecked(Vector3::new(1., 0., 0.)), 0.0);
        for v in front_plane.iter() {
            self.positions.push(rot_mat * v); //+ Vector4::new(0.,10.,0.,0.)
            self.normals.push(rot_mat * fp_norm);
        }
        // right[0]
        let rot_mat =
            Matrix4::<f32>::from_axis_angle(&Unit::new_unchecked(Vector3::new(0., 1., 0.)), PI_2);
        for v in front_plane.iter() {
            self.positions.push(rot_mat * v);
            self.normals.push(rot_mat * fp_norm);
        }
        //left
        let rot_mat =
            Matrix4::<f32>::from_axis_angle(&Unit::new_unchecked(Vector3::new(0., 1., 0.)), -PI_2);
        for v in front_plane.iter() {
            self.positions.push(rot_mat * v);
            self.normals.push(rot_mat * fp_norm);
        }
        //back
        let rot_mat =
            Matrix4::<f32>::from_axis_angle(&Unit::new_unchecked(Vector3::new(0., 1., 0.)), PI);
        for v in front_plane.iter() {
            self.positions.push(rot_mat * v);
            self.normals.push(rot_mat * fp_norm);
        }

        //top
        let rot_mat =
            Matrix4::<f32>::from_axis_angle(&Unit::new_unchecked(Vector3::new(0., 1., 0.)), PI_2);
        let rot_mat_z =
            Matrix4::<f32>::from_axis_angle(&Unit::new_unchecked(Vector3::new(0., 0., 1.)), PI_2);
        for v in front_plane.iter() {
            self.positions.push(rot_mat_z * rot_mat * v);
            self.normals.push(rot_mat_z * rot_mat * fp_norm);
        }

        //bottom
        let rot_mat =
            Matrix4::<f32>::from_axis_angle(&Unit::new_unchecked(Vector3::new(0., 1., 0.)), PI_2);
        let rot_mat_z =
            Matrix4::<f32>::from_axis_angle(&Unit::new_unchecked(Vector3::new(0., 0., 1.)), -PI_2);
        for v in front_plane.iter() {
            self.positions.push(rot_mat_z * rot_mat * v);
            self.normals.push(rot_mat_z * rot_mat * fp_norm);
        }

        //translate along z
        for v in self.positions.iter_mut() {
            v.z += delta_z;
        }

        //generate indexes
        for k in (0..self.positions.len() as u32).step_by(4) {
            self.uvs.push(V::new(1., 1., 0., 0.));
            self.uvs.push(V::new(1., 0., 0., 0.));
            self.uvs.push(V::new(0., 0., 0., 0.));
            self.uvs.push(V::new(0., 1., 0., 0.));

            self.index_buffer.push(k + 0);
            self.index_buffer.push(k + 1);
            self.index_buffer.push(k + 2);

            self.index_buffer.push(k + 0);
            self.index_buffer.push(k + 2);
            self.index_buffer.push(k + 3);
        }

        // println!("Positions:");
        // for p in &self.positions{
        //     println!("{}",p);
        // }
        // println!("END Positions");
    }

    ///Rasterizes the 3d wiremesh to a buffer
    pub fn render<VS, FS>(
        &self,
        stack: &MatrixStack,
        globals: &ShaderGlobals,
        color_buffer: &mut RowBuffer2D<u32>,
        aux_buffers: &mut OutputBuffers,
        vertex_shader: VS,
        fragment_shader: FS,
    ) where
        VS: FnOnce(&mut TriangleData, &ShaderGlobals, &MatrixStack) + Copy,
        FS: FnOnce(
                &(u32, u32),
                &Vector4<f32>,
                &TriangleData,
                &ShaderGlobals,
                &mut RowBuffer2D<u32>,
                &mut OutputBuffers,
            ) ->(u16,u32) + Copy,
    {
        let mut attribs = [TriangleData::new(); 3];
        let mut attrib_mask;

        //loop through 3 triangles at a time and draw
        let mut k = 0;
        let len = self.index_buffer.len();

        while k < len {
            /*copy positions into attribs struct*/
            {
                let attr = attribs[0].get_mut(0);
                for i in 0..3 {
                    attr[i] = self.positions[self.index_buffer[k + i] as usize];
                }
            }

            /*copy normals into attribs struct*/
            {
                let attr = attribs[0].get_mut(1);
                for i in 0..3 {
                    attr[i] = self.normals[self.index_buffer[k + i] as usize];
                }
            }

            /*copy uvs into attribs struct*/
            {
                let attr = attribs[0].get_mut(2);
                for i in 0..3 {
                    attr[i] = self.uvs[self.index_buffer[k + i] as usize];
                }
            }

            //run vertex shader (does model and view transfoms )
            vertex_shader(&mut attribs[0], &globals, &stack);

            //clip triangles ( clips triangle in eye-space )
            let verts_behind_plane_count = geometry::clip_triangle(&mut attribs, -0.2);

            //Use the count from clip_triangle(...) to figure out which attribue is ready to draw
            attrib_mask = match verts_behind_plane_count {
                0 => [true, false, false],
                1 => [false, true, true],
                2 => [false, true, false],
                _ => {
                    k += 3;
                    continue;
                }
            };

            //convert triangles to screen coordinates
            geometry::attribs_to_screen_space(&mut attribs, &mut attrib_mask, &stack);

            //draw valid triangles
            for k in 0..3 {
                if attrib_mask[k] {
                    color_buffer.draw_triangle(&attribs[k], globals, aux_buffers, fragment_shader);
                }
            }
            k += 3;
        }
    }

    pub fn load_model(path: &str) -> Option<Self> {
        let mut model = Self::new();

        let mut positions: Vec<Vector4<f32>> = Vec::new();
        let mut uvs = Vec::new();
        let mut normals = Vec::new();
        let mut vert_counter = 0;

        match File::open(path) {
            Ok(file) => {
                for line in std::io::BufReader::new(file).lines() {
                    let mut token_table: [Option<&str>; 5] = [None; 5];
                    let mut table_size = 0;
                    let data = line.unwrap_or(String::new());
                    let mut data_slice = &data[..];

                    //this just ignores comment lines
                    if data_slice.len() > 0 && &data_slice[0..1] == "#" {
                        continue;
                    }

                    //tokenize space delimited line
                    while let (false, Some(index)) = (data_slice.is_empty(), data_slice.find(" ")) {
                        token_table[table_size] = Some(&data_slice[..index]);
                        table_size += 1;
                        data_slice = &data_slice[index + 1..];
                    }

                    if data_slice.is_empty() == false {
                        token_table[table_size] = Some(&data_slice[..]);
                        table_size += 1;
                    }

                    let prefix = token_table[0].take().unwrap_or(&"");
                    if prefix == "f" {
                        for face_opt in token_table[1..].iter() {
                            if let Some(face_str) = face_opt {
                                let mut iterator = face_str.split('/');

                                let [vert_index, uv_index, norm_index]: [i32; 3] = [
                                    iterator
                                        .next()
                                        .as_ref()
                                        .unwrap_or(&"-1")
                                        .parse()
                                        .unwrap_or(-1)
                                        - 1,
                                    iterator
                                        .next()
                                        .as_ref()
                                        .unwrap_or(&"-1")
                                        .parse()
                                        .unwrap_or(-1)
                                        - 1,
                                    iterator
                                        .next()
                                        .as_ref()
                                        .unwrap_or(&"-1")
                                        .parse()
                                        .unwrap_or(-1)
                                        - 1,
                                ];

                                if vert_index >= 0 {
                                    model.positions.push(positions[vert_index as usize]);
                                }

                                if norm_index >= 0 {
                                    model.normals.push(normals[norm_index as usize]);
                                } else {
                                    model.normals.push(Vector4::zeros());
                                }

                                if uv_index >= 0 {
                                    let vec: Vector4<f32> = uvs[uv_index as usize];
                                    let u = vec.x;
                                    let v = vec.y;
                                    model.uvs.push(Vector4::new(u, v, 0.0, 0.0));
                                } else {
                                    let d = normals[norm_index as usize];
                                    let u = 0.5 + d.x.atan2(d.z) / (2.0 * 3.14159);
                                    let v = 0.5 - d.y.asin() / 3.14159;
                                    model.uvs.push(Vector4::new(u, v, 0.0f32, 0.0f32));
                                }
                                if vert_index >= 0 || norm_index >= 0 || uv_index >= 0 {
                                    model.index_buffer.push(vert_counter as u32);
                                    vert_counter += 1;
                                }
                            }
                        }
                    } else if prefix == "v" || prefix == "vt" || prefix == "vn" {
                        let mut g_vec = Vector4::zeros();
                        for (k, number_opt) in token_table[1..].iter().enumerate() {
                            if let Some(number_str) = number_opt {
                                let comp: f32 = number_str.parse().unwrap_or(0.0);
                                g_vec[k] = comp;
                            }
                        }
                        if prefix == "v" && table_size <= 5 {
                            g_vec[3] = 1.0;
                            positions.push(g_vec);
                        } else if prefix == "vt" {
                            g_vec[2] = 0.;
                            g_vec[3] = 0.;
                            uvs.push(g_vec);
                        } else if prefix == "vn" {
                            g_vec[3] = 0.;
                            normals.push(g_vec);
                        }
                    }
                }

                return Some(model);
            }
            Err(err) => {
                println!("Error: {}", err.description());
            }
        }

        None
    }
}
