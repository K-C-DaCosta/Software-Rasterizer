
#[test]
fn convertion_test(){
    use nalgebra::Vector4; 
    use crate::graphics::{TriangleData,AttribData};

    let mut td = TriangleData::new();
    td.data = [
        //verts
        AttribData { v4_float: Vector4::new(1.,0.,0.,0.)},
        AttribData { v4_float: Vector4::new(0.,1.,0.,0.)},
        AttribData { v4_float: Vector4::new(0.,0.,1.,0.)},
        //normals
        AttribData { v4_float: Vector4::new(0.,0.,0.,0.9)},
        AttribData { v4_float: Vector4::new(1.,0.,0.,0.9)},
        AttribData { v4_float: Vector4::new(0.,7.,0.,0.9)},
        //uvs
        AttribData { v4_float: Vector4::new(0.,0.,1.,0.)},
        AttribData { v4_float: Vector4::new(0.,0.,0.,1.)},
        AttribData { v4_float: Vector4::new(1.,0.,0.,0.)},
        //other?
        AttribData { v4_float: Vector4::new(0.,1.,0.,0.)},
        AttribData { v4_float: Vector4::new(0.,0.,1.,0.)},
        AttribData { v4_float: Vector4::new(0.,0.,0.,1.)},
    ];
    td.convert_to_fixed_point();
    let vert = unsafe{ td.data[5].v4_fixed_norm } ;
    println!("{}",vert);
    panic!("test not implemented"); // no asserts yet. Gotta work this out by hand. 
}