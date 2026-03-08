# compiler
cxx = g++

# compiler flags
cxxFlags = -std=c++17 -O2

# get opencv flags from pkg-config
opencvFlags = $(shell pkg-config --cflags --libs opencv4)

# output binary name
target = camera

# source file
src = camera.cpp

# build target
all: $(target)

# compile camera program
$(target): $(src)
	$(cxx) $(cxxFlags) $(src) -o $(target) $(opencvFlags)

# run camera preview
run: $(target)
	./$(target)

# remove compiled binary
clean:
	rm -f $(target)