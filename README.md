torch-signal
============
Quickstart
----------
Install fftw3 on your OS:
  
OSX (Homebrew):
```bash
brew install fftw
```
  
Ubuntu:
```bash
sudo apt-get install libfftw3
```
  
Install torch-signal:
```bash
luarocks install https://raw.github.com/soumith/torch-signal/master/rocks/signal-scm-1.rockspec
```

(add sudo for ubuntu)

For documentation, go to:
http://soumith.github.io/torch-signal/signal/

For examples, see tests/