LIB=c_model
SRCS=$(wildcard *.cc)
HDRS=$(wildcard *.hh)
default: $(LIB).so

.PHONY: clean

PB11_INC=$(shell python3 -m pybind11 --includes)
PYEXT=$(shell python3-config --extension-suffix)

$(LIB).so: $(SRCS) $(HDRS)
	python3 -m pybind11 --includes
	$(CXX) -g -O0 -Wall -shared -std=c++11 -fPIC $(PB11_INC)  $(SRCS) -o $(LIB)$(PYEXT)

clean:
	rm -f *.so
