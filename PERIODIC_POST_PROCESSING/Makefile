CC = "g++"
CXXFLAGS = -std=c++17
PROJECT = periodic_post_processing

SRC = main.cpp

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(CXXFLAGS) $(SRC) -o $(PROJECT) $(LIBS)

clean :
	rm -rf *.o $(PROJECT)
