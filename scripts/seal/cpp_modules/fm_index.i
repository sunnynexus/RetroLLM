%module fm_index

%{
#include "fm_index.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <typeinfo>
#include <future>
#include <cstdio>
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_pair.i"
%include "std_unordered_map.i"

namespace std {
    %template(IntVector) vector<unsigned long>;
    %template(IntVectorVector) vector<vector<unsigned long>>;
    %template(IntIntPair) pair<unsigned long, unsigned long>;
    %template(StringVectorPair) pair<string, vector<unsigned long>>;
    %template(StringVectorPairVector) vector<pair<string, vector<unsigned long>>>;
    %template(IntIntMap) unordered_map<unsigned long, unsigned long>;
    %template(IntStringMap) unordered_map<string, unsigned long>;
}

%include "fm_index.hpp"

