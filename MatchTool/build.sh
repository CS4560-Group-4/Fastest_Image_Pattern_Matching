set -e

#g++ main.cpp -o ./out/main  -O3 -march=native -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs
clang++ main.cpp -o ./out/main -O3 -fopenmp  -march=native -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs

# Uncomment to run the exe
# LD_LIBRARY_PATH=$LIB_PATH ./out/main

