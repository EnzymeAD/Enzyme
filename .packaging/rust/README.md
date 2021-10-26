# Goal
This repository will build enzyme/llvm/clang/rustc in the right configuration such that you can use it in combination with [oxide-enzyme](https://github.com/rust-ml/oxide-enzyme).

# Requirements
git  
ninja  
cmake  
Rust (rustup)    
~10GB free storage in $HOME/.config  

# Usage
    $ git clone https://github.com/ZuseZ4/enzyme_build  
    $ cd enzyme_build  
    $ cargo run --release  

Depending on your CPU this might take a few hours.  
The build process will run enzyme tests, so your last output should look similar to these lines:

Testing Time: 0.63s  
  Passed           : 240  
  Expectedly Failed:   5  

# Extras
- Q: Can I use some other location to store everyting?
- A: We will add an option for that later. If it's urgent please open an issue or ping me (Manuel Drehwald) on the rust-ml Zulip.  
&nbsp;
- Q: It fails some (all) tests or the build breaks even earlier. Help?
- A: Sorry. Please open an issue with relevant information (OS, error message) or ping me on the rust-ml Zulip.  
&nbsp;
- Q: How often do I have to run this? It takes quite a while..
- A: We are aware of this and working on offering pre-build versions. In the meantime you have to build it once per rust Version. So we will publish an update once 1.57 hits stable.
