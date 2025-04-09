set -e

INCLUDE_PATH=/usr/local/include/opencv4
LIB_PATH=/usr/local/lib
# clang++ main.cpp -g -O3 -o ./out/main -fopenmp=libiomp5 -I$INCLUDE_PATH -L$LIB_PATH -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs
g++ main.cpp -march=native -g -O3 -o ./out/main -fopenmp -I$INCLUDE_PATH -L$LIB_PATH -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs

# Uncomment to run the exe
# LD_LIBRARY_PATH=$LIB_PATH ./out/main tests/10/Src10.bmp tests/10/Dst10.jpg

