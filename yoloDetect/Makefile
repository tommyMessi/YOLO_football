CC = g++
#VPATH = .
#SRCS = $(wildcard *.cpp)  $(wildcard *.c)

INC =  -L /usr/local/cuda/lib64
cflags = -I ../src -I /usr/local/cuda/include

SRCS = yolo_console_dll.cpp
OBJS = *.o 

#INC = -L ../Bin -L /usr/local/cuda/lib64
#cflags = -I ../src -I ../yoloDetect -I /usr/local/cuda/include

LIBRARY = yolo-test

C_FLAGS = -g -w -lpthread -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_legacy  -lcuda -lcudart -lcublas -lcurand ../libdarknet.so


${LIBRARY}: ${SRCS}
	$(CC) $(SRCS) $(C_FLAGS) -o $(LIBRARY) $(INC) $(cflags)
	cp $(LIBRARY) ../

clean:
	rm  $(LIBRARY) 

	
