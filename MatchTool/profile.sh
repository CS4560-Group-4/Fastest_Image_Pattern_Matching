perf record -F 99 -a -g -- ./out/main tests/10/Src10.bmp tests/10/Dst10.jpg
perf script > out.perf 
~/FlameGraph/stackcollapse-perf.pl out.perf > out.folded