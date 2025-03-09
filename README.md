# Installation and Running

This project uses OpenCV, which needs to be compiled from source.
You can look at the `Dockerfile` to see how to compile it. 
(NOTE that actually running the dockerfile DOES NOT work out of the box since GTK+ does not work with docker by default)

After you have compiled and installed OpenCV look at `./MatchTool/build.sh`

Make sure that `INCLUDE_PATH` and `LIB_PATH` are point to the correct files in your system.
(Usually OpenCV install at `/usr/local/include/opencv4` for include and `/usr/local/lib` for libs)
After that run `./build.sh` to compile the test binary (Make sure you are in the `./MatchTool` directory)
Finally uncomment the last line in `build.sh` and run it again or directly run `LD_LIBRARY_PATH=$LIB_PATH ./out/main`.

# Getting Docker to work with MatchTool

This only works if you use X11 (probably not going to work on WSL)

Build the dockerfile:

```
docker build -t foo .  
```

Run the following command to do some X11 client-server magic:

```
xhost +"local:docker@" 
```

Run the built docker image with some extra settings

```
docker run -it --net=host -e DISPLAY=$DISPLAY foo
```


