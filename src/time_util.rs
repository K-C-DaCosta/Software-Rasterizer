use std::time::{Duration,Instant};
use std::fmt; 

#[derive(Copy,Clone)]
pub struct StopWatch{
    min:u128, 
    max:u128, 
    avg:f64, 
    avg_n:f64,
    t0:Option<Instant>,
}

impl StopWatch{
    pub fn new()->StopWatch{
        StopWatch{
            min:999999999,
            max:0,
            avg:0.0,
            avg_n:1.0,
            t0:None,
        }
    }

    pub fn start(&mut self){
        self.t0 = Some(Instant::now());
    }

    pub fn elapsed(&self)->Duration{
        self.t0.unwrap().elapsed()
    }

    pub fn stop(&mut self){
        //calculate elapsed time in milliseconds 
        let dt = self.t0.unwrap().elapsed().as_millis();
        //update metrics 
        self.min = self.min.min(dt);
        self.max = self.max.max(dt);
        self.avg+= (1.0/self.avg_n)*(dt as f64 - self.avg);
        self.avg_n+=1.0; 
    }
    pub fn reset(&mut self){
        self.min = 999999999; 
        self.max = 0; 
        self.avg = 0.0; 
        self.avg_n = 1.0; 
        self.t0 = None; 
    }
}

impl fmt::Display for StopWatch{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[min:{:03},max:{:03},avg:{:.3}]",self.min,self.max,self.avg)
    }
}

pub struct Timer{ 
    t0:Option<Instant>,
}

impl Timer{
    pub fn new()->Timer{
        Timer{
            t0:Some(Instant::now()), 
        }
    }
    pub fn execute<F:FnMut()>(&mut self, dt:u128, mut task:F){
        if self.t0.unwrap().elapsed().as_millis() > dt {
            task();
            self.t0 = Some(Instant::now())
        } 
    }
}