# What is this? 
A Software rasterizer written in pure rust. Im not really finished with it just yet, but if one wished, they could use it as an OBJ viewer or even use it as a backend in a graphics lib. 

#  What you need
You need to install `Cargo` and have `SDL2` available on your system.\
For cargo just go here: 
`https://www.rust-lang.org/tools/install` <br/>

To get `SDL2` on Ubuntu open your terminal enter:
```
    sudo apt install libsdl2 libsdl2-dev
```
Pretty much every linux distro has SDL2 available in their repositories. So just check your package manager, and make sure SDL2 is installed. 

# How to run

```
cargo run --release 
```

# Controls
- Press and hold shift to move camera faster 
- Use the mouse keys to change camera orientations 
- Use WASD keys to move around in space 

#  Other notes
I Still need to add multithreading in order to take full advantage of 
multiprocessor computers.  Im thinking either raylib or I roll my own 
solution. 

# Screenshots
<img src="./screenshots/Screenshot%20from%202020-05-05%2013-12-56.png"/>
<img src="./screenshots/Screenshot%20from%202020-05-05%2013-12-47.png"/>
