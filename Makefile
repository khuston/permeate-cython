NUMPY_INC=/usr/local/lib/python2.7/site-packages/numpy/core/include/
PYTHON_INC=/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Headers/
PYTHON_LIB=/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/Current/lib/

all:
	python setup.py build_ext --inplace -I$(PYTHON_INC) -I$(NUMPY_INC) -L$(PYTHON_LIB) -lpython2.7
clean:
	rm *.so
