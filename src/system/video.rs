use super::super::image::buffer::RowBuffer2D;
use super::super::image::Buffer2D;

use sdl2::event::*;

use sdl2::keyboard::Keycode;

use sdl2::pixels::{Color, PixelFormatEnum};

use sdl2::rect::Rect;

use sdl2::render::{Canvas, Texture, TextureAccess};
use sdl2::video;
use sdl2::video::{DisplayMode, WindowSurfaceRef};

use std::cell::RefCell;
use std::rc::Rc;

use std::slice;
use std::ptr;
use std::mem;
use std::marker; 

pub fn ref_to_mut_ptr<T>(a:&mut T)->*mut T{
    a as *mut T
}

pub fn ref_to_ptr<T>(a:&T)->*const T{
     a as *const T
}


pub struct Window {
    width: u32,
    height: u32,
    title: String,
    event_pump: Option<sdl2::EventPump>,
    window_buffer: RowBuffer2D<u32>,
    pub context: sdl2::Sdl,
    pub sdl_window: Option<video::Window>,
}

pub struct EventPumpRef{
    event_pump:*mut sdl2::EventPump,
}

impl EventPumpRef{
    pub fn  as_mut(& mut self)-> & mut sdl2::EventPump{
        unsafe{
            &mut *self.event_pump
        }
    }
}

impl Window {
    pub fn new(width: u32, height: u32, title: String) -> Window {
        let mut win = Window {
            width,
            height,
            title,
            event_pump: None,
            window_buffer: RowBuffer2D::new(),
            context: sdl2::init().unwrap(),
            sdl_window: None,
        };

        //init window
        let mut sdl_window = win
            .context
            .video()
            .unwrap()
            .window(win.title.as_str(), win.width, win.height)
            .resizable()
            .build()
            .unwrap();

        //ensure that the window is ideal format 
        sdl_window
            .set_display_mode(DisplayMode::new(
                PixelFormatEnum::ARGB8888,
                width as i32,
                height as i32,
                60,
            ))
            .unwrap();


        win.sdl_window = Some(sdl_window);

        //setup event pump
        win.event_pump = Some(win.context.event_pump().unwrap());
        win.window_buffer.allocate(width, height, 0);
        win
    }

    pub fn get_event_pump(& mut self)->EventPumpRef{
        EventPumpRef{
            event_pump:ref_to_mut_ptr(self.event_pump.as_mut().unwrap()),
    
        }
    }
    
    pub fn buffer(& mut self)->& mut RowBuffer2D<u32>{
        &mut self.window_buffer
    }

    pub fn resize(&mut self, mut w:i32,mut h:i32){
        w = w.max(0);
        h = h.max(0);
        self.width = w as u32 ; 
        self.height = h as u32 ; 
        self.window_buffer.resize(w, h);
    }

    pub fn get_surface(&self) -> WindowSurfaceRef {
        let event_pump_ref = self.event_pump.as_ref().unwrap();
        self.sdl_window
            .as_ref()
            .unwrap()
            .surface(event_pump_ref)
            .unwrap()
    }

    pub fn update(&mut self) {
        let window_buffer_ptr = ref_to_mut_ptr(self.buffer());
        let mut surface = self.get_surface();
        let pixels = surface.without_lock_mut().unwrap();
        unsafe{
            let window_buffer= &mut *window_buffer_ptr;
            ptr::copy_nonoverlapping(window_buffer.as_bytes_mut().as_mut_ptr(),pixels.as_mut_ptr(),pixels.len());
        }
        self.get_surface().finish().unwrap();
    }

    pub fn dimensions(&self)->(u32,u32){
        (self.width,self.height)
    }

    pub fn width(&self)->u32{
        self.width
    }
    
    pub fn height(&self)->u32{
        self.height
    }
}
