PYTHON_INCLUDE="$(python3 -c 'from sysconfig import get_paths;print(get_paths()["include"])')"
g++ -fPIC -c mlpcppwrapper.cpp -o mlpcppwrapper.o
swig -c++ -python -o mlpcppwrapper_wrap.cxx mlpcppwrapper.i
g++ -fPIC -I$PYTHON_INCLUDE -c mlpcppwrapper_wrap.cxx -o mlpcppwrapper_wrap.o
g++ -Xlinker -export-dynamic -shared mlpcppwrapper.o mlpcppwrapper_wrap.o  -o _mlpcppwrapper.so
