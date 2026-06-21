GUROBI_INSTALL_DIR = /opt/gurobi1302/linux64
INC      = $(GUROBI_INSTALL_DIR)/include
CPP      = g++
CPPARGS  = -std=c++11 -m64
CPPLIB   = -L$(GUROBI_INSTALL_DIR)/lib -lgurobi_c++ -lgurobi130

guroback: guroback.cpp
	$(CPP) $(CPPARGS) -o $@ $< -I$(INC) $(CPPLIB) -lm -O2 -Wall -Wextra -Wpedantic -Werror
