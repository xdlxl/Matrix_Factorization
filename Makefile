CC = g++
CPPFLAGS = -Wall -g -O3 -fPIC -std=c++11 -march=native 
INCLUDES = -I. -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux
LDFLAGS = -L. -L/usr/lib/jvm/java-1.6.0-openjdk-1.6.0.34.x86_64/jre/lib/amd64/server/ -pthread -lz -ljvm -lhdfs

all: mf_train mf_predict 

#.cpp.o:
#	$(CC) -c $^ $(INCLUDES) $(CPPFLAGS)
src/mf_train.o: src/mf_train.cpp src/*.h 
	$(CC) -c src/mf_train.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

src/mf_predict.o: src/mf_predict.cpp src/*.h
	$(CC) -c src/mf_predict.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

src/stopwatch.o: src/stopwatch.cpp src/stopwatch.h
	$(CC) -c src/stopwatch.cpp -o $@ $(INCLUDES) $(CPPFLAGS)

mf_train: src/mf_train.o src/stopwatch.o 
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

mf_predict: src/mf_predict.o src/stopwatch.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -f src/*.o mf_train mf_predict
