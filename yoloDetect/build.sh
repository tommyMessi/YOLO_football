rm libyD.so

#INC= -L ../Bin -L /usr/local/cuda-7.5/lib64
g++ -g -O3 -fPIC *.cpp *.hpp -shared -o libyD.so  -I ./ -I ../src -L ../Bin ../Bin/libtest.a -L ../Bin -L /usr/local/cuda-7.5/lib64
cp  libyD.so ./../Bin/
#rm  libveCarTypeReco.so
