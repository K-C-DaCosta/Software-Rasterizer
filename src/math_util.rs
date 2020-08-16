use std::collections::HashMap;
use nalgebra::*;

pub mod fixed_point;
pub mod fixed_vector;
pub mod geometry; 
pub mod math_tests; 

type Mat4f = Matrix4<f32>; 


pub struct Camera{
    pub p:Vector4<f32>, 
    pub q:Vector4<f32>, 
    pub r:Vector4<f32>, 
    pub o:Vector4<f32>,
}

impl Camera{
    
    pub fn new()->Camera{
        Camera{
            p:Vector4::new(1.,0.,0.,0.),
            q:Vector4::new(0.,1.,0.,0.),
            r:Vector4::new(0.,0.,1.,0.),
            o:Vector4::new(0.,0.,0.,1.),
        }
    }

    pub fn update(&mut self,up:&Vector4<f32>){
        //compute new basis vectors
        self.p = up.xyz().cross(&self.r.xyz()).to_homogeneous();
        self.q = self.r.xyz().cross(&self.p.xyz()).to_homogeneous();

        //normalize all basis vectors 
        self.p = self.p.normalize();
        self.q = self.q.normalize();
        self.r = self.r.normalize();
    }

    pub fn compute_view_mat(&self )->Matrix4<f32>{
        let (p,q,r,o) = (&self.p,&self.q,&self.r,&self.o);
        Matrix4::new(
            p.x,p.y,p.z,-p.dot(o),
            q.x,q.y,q.z,-q.dot(o),
            r.x,r.y,r.z,-r.dot(o),
            0.0,0.0,0.0,   1.0   ,
        )
    }
    ///translates camera by checking if WASD were pressed 
    pub fn generic_keyboard_handling(&mut self, key_map:&HashMap<i32,bool>, stack:&mut MatrixStack){
        use sdl2::keyboard::Keycode;
     
        let camera = self;
        
        let mut speed = 0.01; 
        //if shift is pushed move faster
        if let Some(true) = key_map.get(&(Keycode::LShift as i32)) {
             speed = 0.1;
        }

        if let Some(true) = key_map.get(&(Keycode::W as i32)) {
            camera.o -= camera.r * speed;
            stack.view = camera.compute_view_mat();
        }
        
        if let Some(true) = key_map.get(&(Keycode::A as i32)) {
            camera.o -= camera.p * speed;
            stack.view = camera.compute_view_mat();
        }

        if let Some(true) = key_map.get(&(Keycode::S as i32)) {
            camera.o += camera.r * speed;
            stack.view = camera.compute_view_mat();
        }

        if let Some(true) = key_map.get(&(Keycode::D as i32)) {
            camera.o += camera.p * speed;
            stack.view = camera.compute_view_mat();
        }
    }
}


pub struct MatrixStack{
    pub viewport:Mat4f,
    pub projection:Mat4f,
    pub view:Mat4f,
    pub modelview:Mat4f, 
    pub normalmat:Mat4f,
    pub stack:Vec<Mat4f>, 
}



impl MatrixStack{
    pub fn new()->MatrixStack{
        let _ortho = Orthographic3::new(-100., 100., -50., 50., 1., 10000.).to_homogeneous();
        let _persp = Perspective3::new(16.0/9.0, 3.14159/3.0, 1., 10000.).to_homogeneous();
        MatrixStack{
            viewport:Matrix4::identity(),
            projection: _persp, 
            view:Matrix4::identity(),
            modelview:Matrix4::identity(),
            normalmat:Matrix4::identity(),
            stack:vec![Matrix4::identity()],
        }
    }

    pub fn push(&mut self,m:Mat4f){
        self.stack.push(
            m*self.peek()
        );
    }

    pub fn push_rotate(&mut self, axis:Vector3<f32>, angle:f32){
        let unit_axis  = &Unit::new_normalize(axis);
        let rot = Rotation3::from_axis_angle(unit_axis,angle).to_homogeneous(); 
        self.push(rot);
    }
    pub fn push_translate(&mut self,disp:Vector3<f32>){
        let trans = Translation3::from(disp).to_homogeneous();
        self.push(trans);
    }

    pub fn pop(&mut self){
        if self.stack.len() > 1 {
            self.stack.pop();
        }
    }

    pub fn peek(&self)->&Mat4f{
        self.stack.last().as_ref().unwrap()
    }

    //updates modelview matrix
    pub fn update(&mut self){
        self.modelview = self.view*self.peek();
        self.normalmat = self
            .modelview
            .try_inverse()
            .unwrap_or( Mat4f::identity() )
            .transpose();
    }

    pub fn update_viewport(&mut self, w:u32,h:u32){
        let hw = w as f32 * 0.5; 
        let hh = h as f32 * 0.5; 
        self.viewport = Matrix4::new(
            hw,0. ,0.,hw,
            0.,-hh ,0.,hh,
            0.,0. ,0.,0.,
            0.,0. ,0.,1.
        );
    }
}