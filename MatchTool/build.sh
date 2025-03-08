set -e
g++ main.cpp -o ./out/main -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs
LD_LIBRARY_PATH=/usr/local/lib ./out/main

