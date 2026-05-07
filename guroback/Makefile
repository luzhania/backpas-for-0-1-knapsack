GUROBI_INSTALL_DIR = ~/gurobi1103/linux64
INC      = $(GUROBI_INSTALL_DIR)/include
CPP      = g++
CPPARGS  = -std=c++11 -m64
CPPLIB   = -L$(GUROBI_INSTALL_DIR)/lib  -lgurobi110 -lgurobi_c++

guroback: guroback.cpp
	$(CPP) $(CPPARGS) -o $@ $< -I$(INC) $(CPPLIB) -lm -O2 -Wall -Wextra -Wpedantic -Werror
