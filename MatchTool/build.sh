set -e

INCLUDE_PATH=/usr/local/include/opencv4
LIB_PATH=/usr/local/lib
g++ main.cpp -o ./out/main -I$INCLUDE_PATH -L$LIB_PATH -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs

# Uncomment to run the exe
# LD_LIBRARY_PATH=$LIB_PATH ./out/main

