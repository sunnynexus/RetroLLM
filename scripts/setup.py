from distutils.core import setup, Extension
import os

extra_compile_args = ["-std=c++11", "-DNDEBUG", "-O3"]

extension = Extension(
    "seal.cpp_modules._fm_index",
    include_dirs=["seal/cpp_modules", os.path.expanduser("~/include")],
    libraries=["stdc++", "sdsl", "divsufsort", "divsufsort64", "pthread"],
    library_dirs=[os.path.expanduser("~/lib")],
    sources=["seal/cpp_modules/fm_index.cpp", "seal/cpp_modules/fm_index.i"],
    swig_opts=["-I../include", "-c++"],
    language="c++11",
    extra_compile_args=extra_compile_args,
)

setup(
    name="SEAL",
    version="1.0",
    ext_modules=[extension],
)

"""
SWIG安装
wget http://prdownloads.sourceforge.net/swig/swig-4.0.2.tar.gz
tar zxvf swig-4.0.2.tar.gz

cd swig-4.2.1
./configure --without-pcre --prefix=/home/u2023000153
make -j
make install

cd RetroLLM
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' scripts/res/external/sdsl-lite/install.sh

swig -c\+\+ -python scripts/seal/cpp_modules/fm_index.i  &&  python setup.py build_ext --inplace

"""
