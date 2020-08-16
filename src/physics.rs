#![allow(dead_code)]

use super::graphics::Graphics;
use super::image::buffer::RowBuffer2D;
use nalgebra::Vector2;

/*
Note:
This is an unfinished  attempt at implementing a flip-fluid simulator. It currently doesn't work.
*/
pub struct BilearWeights {
    pub w00: f32,
    pub w01: f32,
    pub w10: f32,
    pub w11: f32,
}
pub struct Particle {
    pub pos: Vector2<f32>,
    pub vel: Vector2<f32>,
}

impl Particle {
    pub fn new(pos: Vector2<f32>, vel: Vector2<f32>) -> Particle {
        Particle { pos, vel }
    }
}

pub struct FlipGrid {
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub weight_cache: Vec<f32>,
    pub v1: Vec<f32>,
    pub pressure: Vec<f32>,
    pub divergence: Vec<f32>,
    pub w: i32,
    pub h: i32,
    pub cell_len: f32, //physical length of cell
}

impl FlipGrid {
    pub fn new(w: i32, h: i32) -> FlipGrid {
        let size = (w + 2) * (h + 2);
        FlipGrid {
            u: (0..size).map(|_| 0.0).collect::<Vec<_>>(),
            v: (0..size).map(|_| 0.0).collect::<Vec<_>>(),
            weight_cache: (0..size).map(|_| 0.0).collect::<Vec<_>>(),
            v1: (0..size).map(|_| 0.0).collect::<Vec<_>>(),
            pressure: (0..size).map(|_| 0.0).collect::<Vec<_>>(),
            divergence: (0..size).map(|_| 0.0).collect::<Vec<_>>(),
            w,
            h,
            cell_len: 1.0,
        }
    }

    pub fn compute_divergence(&mut self) {
        let (w, h) = self.get_dim();
        for i in 1..=h {
            for j in 1..=w {
                self.divergence[Self::id(w, i, j)] = -(self.u[Self::id(w, i, j + 1)]
                    - self.u[Self::id(w, i, j - 1)]
                    + self.v[Self::id(w, i + 1, j)]
                    - self.v[Self::id(w, i - 1, j)]);
            }
        }
    }
    pub fn compute_pressure(&mut self) {
        let (w, h) = self.get_dim();

        for _ in 0..20 {
            for i in 1..=h {
                for j in 1..=w {
                    self.pressure[Self::id(w, i, j)] = (self.pressure[Self::id(w, i + 1, j)]
                        - self.pressure[Self::id(w, i - 1, j)]
                        + self.pressure[Self::id(w, i, j + 1)]
                        - self.pressure[Self::id(w, i, j - 1)]
                        + self.divergence[Self::id(w, i, j)])
                        / 6.0;
                }
            }
        }
    }
    pub fn compute_weights_aux(p: &Particle, inv_cs: f32) -> (f32, f32, BilearWeights) {
        let i = ((p.pos.y * inv_cs).floor() + 1.).max(0.);
        let j = ((p.pos.x * inv_cs).floor() + 1.).max(0.);
        let weights = Self::compute_weights(&p.pos, inv_cs, i, j, 0.0, 0.0);
        (i, j, weights)
    }
    ///pub sample velocity
    pub fn interpolate_velocity(&mut self, aux_weights: (f32, f32, BilearWeights)) -> Vector2<f32> {
        let (w, _) = self.get_dim();
        let i = aux_weights.0 as i32;
        let j = aux_weights.1 as i32;
        let weights = aux_weights.2;

        let c00 = 1.0 / self.weight_cache[Self::id(w, i + 0, j + 0)];
        let c10 = 1.0 / self.weight_cache[Self::id(w, i + 0, j + 1)];
        let c01 = 1.0 / self.weight_cache[Self::id(w, i + 1, j + 0)];
        let c11 = 1.0 / self.weight_cache[Self::id(w, i + 1, j + 1)];

        let u00 = self.u[Self::id(w, i + 0, j + 0)];
        let u10 = self.u[Self::id(w, i + 0, j + 1)];
        let u01 = self.u[Self::id(w, i + 1, j + 0)];
        let u11 = self.u[Self::id(w, i + 1, j + 1)];
        let v00 = self.v[Self::id(w, i + 0, j + 0)];
        let v10 = self.v[Self::id(w, i + 0, j + 1)];
        let v01 = self.v[Self::id(w, i + 1, j + 0)];
        let v11 = self.v[Self::id(w, i + 1, j + 1)];

        let u = weights.w00 * u00 * c00
            + weights.w10 * u10 * c10
            + weights.w01 * u01 * c01
            + weights.w11 * u11 * c11;
        let v = weights.w00 * v00 * c00
            + weights.w10 * v10 * c10
            + weights.w01 * v01 * c01
            + weights.w11 * v11 * c11;
        Vector2::new(u, v)
    }
    ///interpolates velocity into grid
    #[allow(unused_variables)]
    pub fn transfer(&mut self, p: &Particle, inv_cs: f32, aux_weights: (f32, f32, BilearWeights)) {
        let (w, _) = self.get_dim();

        let i = aux_weights.0 as i32;
        let j = aux_weights.1 as i32;
        let weights = aux_weights.2;

        self.weight_cache[Self::id(w, i + 0, j + 0)] += weights.w00;
        self.weight_cache[Self::id(w, i + 0, j + 1)] += weights.w10;
        self.weight_cache[Self::id(w, i + 1, j + 0)] += weights.w01;
        self.weight_cache[Self::id(w, i + 1, j + 1)] += weights.w11;

        let i = i as i32;
        let j = j as i32;

        //linearly transfer velocity to grid
        self.u[Self::id(w, i + 0, j + 0)] += weights.w00 * p.vel.x;
        self.v[Self::id(w, i + 0, j + 0)] += weights.w00 * p.vel.y;
        self.u[Self::id(w, i + 0, j + 1)] += weights.w10 * p.vel.x;
        self.v[Self::id(w, i + 0, j + 1)] += weights.w10 * p.vel.y;
        self.u[Self::id(w, i + 1, j + 0)] += weights.w01 * p.vel.x;
        self.v[Self::id(w, i + 1, j + 0)] += weights.w01 * p.vel.y;
        self.u[Self::id(w, i + 1, j + 1)] += weights.w11 * p.vel.x;
        self.v[Self::id(w, i + 1, j + 1)] += weights.w11 * p.vel.y;
    }

    #[allow(unused_variables)]
    pub fn compute_weights(
        pos: &Vector2<f32>,
        inv_cs: f32,
        i: f32,
        j: f32,
        i_off: f32,
        j_off: f32,
    ) -> BilearWeights {
        let u = (pos.x * inv_cs).fract();
        let v = (pos.y * inv_cs).fract();
        let w00 = (1.0 - u) * (1.0 - v);
        let w10 = u * (1.0 - v);
        let w01 = (1.0 - u) * v;
        let w11 = u * v;
        // println!("{},{},{},{},u={},v{}",w00,w10,w01,w11,u,v);
        BilearWeights { w00, w10, w01, w11 }
    }

    pub fn project(&mut self) {
        let (w, h) = self.get_dim();

        self.compute_divergence();
        self.compute_pressure();

        //subtract gradient from divergent velocity field to yield
        //a divergent-free velocity field
        for i in 1..=h {
            for j in 1..=w {
                let dpdx =
                    self.pressure[Self::id(w, i, j + 1)] - self.pressure[Self::id(w, i, j - 1)];
                let dpdy =
                    self.pressure[Self::id(w, i + 1, j)] - self.pressure[Self::id(w, i - 1, j)];
                self.u[Self::id(w, i, j)] -= dpdx * 0.5;
                self.v[Self::id(w, i, j)] -= dpdy * 0.5;
            }
        }
    }

    pub fn enforce_velocity_boundaries(&mut self) {
        let (w, h) = self.get_dim();

        //loop down columns of the grid and reflect velocity
        for i in 1..=h {
            let ln = Vector2::new(1.0, 0.0);
            let rn = ln * -1.0;

            let l_v = Vector2::new(self.u[Self::id(w, i, 1)], self.v[Self::id(w, i, 1)]);

            let r = l_v - ln * ln.dot(&l_v);
            self.u[Self::id(w, i, 1)] = r.x;
            self.v[Self::id(w, i, 1)] = r.y;

            let r_v = Vector2::new(self.u[Self::id(w, i, w)], self.v[Self::id(w, i, w)]);

            let r = r_v - rn * rn.dot(&r_v);
            self.u[Self::id(w, i, w)] = r.x;
            self.v[Self::id(w, i, w)] = r.y;
        }

        //loop down columns of the grid and reflect velocity
        for j in 1..=w {
            let tn = Vector2::new(0.0, 1.0);
            let bn = tn * -1.0;

            let l_v = Vector2::new(self.u[Self::id(w, 1, j)], self.v[Self::id(w, 1, j)]);

            let r = l_v - tn * tn.dot(&l_v);
            self.u[Self::id(w, 1, j)] = r.x;
            self.v[Self::id(w, 1, j)] = r.y;

            let r_v = Vector2::new(self.u[Self::id(w, h, j)], self.v[Self::id(w, h, j)]);

            let r = r_v - bn * bn.dot(&r_v);
            self.u[Self::id(w, h, j)] = r.x;
            self.v[Self::id(w, h, j)] = r.y;
        }
    }

    pub fn draw_grid(&mut self, buffer: &mut RowBuffer2D<u32>, cell_len: i32) {
        let (w, h) = self.get_dim();

        let p0 = self.pressure[Self::id(w, 1, 1)];
        let mut min = p0;
        let mut max = p0;

        self.pressure.iter().for_each(|p| {
            min = p.min(min);
            max = p.max(max);
        });

        for i in 1..=h {
            for j in 1..=w {
                let x = (j - 1) * cell_len;
                let y = (i - 1) * cell_len;
                //bordered cell
                let pressure_real = (self.pressure[Self::id(w, i, j)] - min) / (max - min);
                let mut pressure_color = (pressure_real * 255.0) as u32;
                //clamp color between 0-255
                pressure_color = pressure_color.min(255).max(0);

                buffer.draw_rect(x, y, cell_len + 1, cell_len + 1, 0x00ff0000);
                buffer.draw_rect(x + 1, y + 1, cell_len - 1, cell_len - 1, pressure_color);
            }
        }

        for i in 1..=h {
            for j in 1..=w {
                let x = (j - 1) * (cell_len);
                let y = (i - 1) * (cell_len);
                let center = Vector2::new(x as f32, y as f32);
                let vel = Vector2::new(self.u[Self::id(w, i, j)], self.v[Self::id(w, i, j)]);
                let mag = (self.weight_cache[Self::id(w, i, j)] * 10.0) as i32;
                buffer.draw_rect(
                    center.x as i32 - (mag + 1) / 2,
                    center.y as i32 - (mag + 1) / 2,
                    mag + 1,
                    mag + 1,
                    0x0000ff00,
                );
                buffer.draw_line(center, center + vel * 10.0, 0x0000ff0f);
            }
        }
    }

    pub fn get_dim(&self) -> (i32, i32) {
        (self.w, self.h)
    }

    fn id(w: i32, i: i32, j: i32) -> usize {
        (i * (w + 2) + j) as usize
    }
}

#[allow(dead_code, unused_variables, unused_mut)]
fn flip_test() {
    let mut particles: Vec<Particle> = Vec::new();
    let mut grid = FlipGrid::new(8, 8);

    let cell_size = 100.0;
    let inv_cs = 1.0 / cell_size;

    // grid.transfer(&p, 1.0/cell_size);
    for i in 1..grid.h {
        for j in 1..=grid.w {
            grid.u[FlipGrid::id(grid.w, i, j)] = rand::random();
            grid.v[FlipGrid::id(grid.w, i, j)] = rand::random();
        }
    }
    // let bounds = AABB2D {
    //     tl: Vector2::new(x_pos as i32 - 128, y_pos as i32),
    //     br: Vector2::new(x_pos as i32 + 256 - 128, y_pos as i32 + 256),
    // };
    // texture.render_image(window.buffer(), bounds, LinearSampler2D);

    // let gw = cell_size * grid.w as f32;
    // let gh = cell_size * grid.h as f32;

    // let steps = 5;
    // let dt = 1.0/(steps as f32);
    // for _ in 0..steps{

    //     //clear grid
    //     for ((u, v), w) in grid
    //         .u
    //         .iter_mut()
    //         .zip(grid.v.iter_mut())
    //         .zip(grid.weight_cache.iter_mut())
    //     {
    //         *u = 0.;
    //         *v = 0.;
    //         *w = 0.;
    //     }

    //     for p in particles.iter_mut() {
    //         if p.pos.x < 0.0 {
    //             p.vel.x += -p.pos.x;
    //             p.pos.x = 0.0;
    //         }
    //         if p.pos.x > gw {
    //             p.vel.x += gw - p.pos.x;
    //             p.pos.x = gw - 1.;
    //         }

    //         if p.pos.y < 0.0 {
    //             p.vel.y += -p.pos.y;
    //             p.pos.y = 0.0;
    //         }
    //         if p.pos.y > gh - 10. {
    //             p.vel.y += gh - 10. - p.pos.y;
    //             p.pos.y = gh - 10.;
    //         }
    //     }

    //     //transfer PARTICLE->GRID
    //     for p in particles.iter_mut() {
    //         let aux_weights = FlipGrid::compute_weights_aux(p, inv_cs);
    //         grid.transfer(&p, inv_cs, aux_weights);
    //     }

    //     //enforce incompressibility
    //     grid.enforce_velocity_boundaries();
    //     grid.project();

    //     for p in particles.iter_mut() {
    //         let aux_weights = FlipGrid::compute_weights_aux(p, inv_cs);
    //         //transfer GRID->PARTICLE
    //         //p.vel = grid.interpolate_velocity(aux_weights);
    //         //move particle forward in time [*fixed timestep|*dt=1|*no substeps]
    //         p.pos += p.vel*dt;
    //         // p.vel+=Vector2::new(0.,1.0);
    //     }
    // }

    // grid.draw_grid(window.buffer(), cell_size as i32);
    // for p in particles.iter() {
    //     window
    //         .buffer()
    //         .draw_rect(p.pos.x as i32 - 5, p.pos.y as i32, 5, 5, 0xff);
    // }
}
