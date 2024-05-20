CC = "g++"
CXXFLAGS = -std=c++17
PROJECT = periodic_mission_post_processing

SRC = periodic_mission_post_processing.cpp

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(CXXFLAGS) $(SRC) -o $(PROJECT) $(LIBS)

clean :
	rm -rf *.o $(PROJECT)
