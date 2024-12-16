#include "fm_index.hpp"

#include <sdsl/suffix_arrays.hpp>
#include <sdsl/io.hpp>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <cstdio>

#include <unordered_map>
#include <string>
#include <algorithm>
// #include <experimental/filesystem>  // For C++14
// namespace fs = std::experimental::filesystem;
// #include <filesystem>  // For C++17
// namespace fs = std::filesystem;

using namespace sdsl;
using namespace std;

typedef csa_wt_int<> fm_index_type;
typedef unsigned long size_type;

size_type SHIFT = 10;


FMIndex::FMIndex() {
    query_ = int_vector<>(4096);
}

FMIndex::~FMIndex() {}

void FMIndex::initialize(const vector<size_type> &data) {
    int_vector<> data2 = int_vector<>(data.size());
    for (size_type i = 0; i < data.size(); i++) data2[i] = data[i];
    construct_im(index, data2, 0);
    chars = vector<size_type>(index.wavelet_tree.sigma);
    rank_c_i = vector<size_type>(index.wavelet_tree.sigma);
    rank_c_j = vector<size_type>(index.wavelet_tree.sigma);
    occurring_distinct = distinct(0, this->size());
    // cout << "occurring_distinct: " << occurring_distinct << endl;
}

void FMIndex::initialize_from_file(const string file, int width) {
    construct(index, file, width);
    chars = vector<size_type>(index.wavelet_tree.sigma);
    rank_c_i = vector<size_type>(index.wavelet_tree.sigma);
    rank_c_j = vector<size_type>(index.wavelet_tree.sigma);
}

size_type FMIndex::size() {
    return index.size();
}


const vector<size_type> FMIndex::backward_search_multi(const vector<size_type> query)
{
    vector<size_type> output;
    size_type l = 0;
    size_type r = index.size();
    for (size_type i = 0; i < query.size(); i++)
        backward_search(index, l, r, (size_type) query[i], l, r);
    output.push_back(l);
    output.push_back(r+1);
    return output;
}

const vector<size_type> FMIndex::backward_search_step(size_type symbol, size_type low, size_type high) 
{
    vector<size_type> output;
    size_type new_low = 0;
    size_type new_high = 0;
    backward_search(index, low, high, (size_type) symbol, new_low, new_high);
    output.push_back(new_low);
    output.push_back(new_high);
    return output;
}

const vector<size_type> FMIndex::distinct(size_type low, size_type high) 
{
    vector<size_type> ret;
    if (low == high) return ret;
    size_type quantity;                          // quantity of characters in interval
    interval_symbols(index.wavelet_tree, low, high, quantity, chars, rank_c_i, rank_c_j);
    for (size_type i = 0; i < quantity; i++)
    { 
        ret.push_back(chars[i]);
    }
    return ret; 
}

const vector<size_type> FMIndex::distinct_count(size_type low, size_type high) 
{

    vector<size_type> chars_ = vector<size_type>(index.wavelet_tree.sigma);
    vector<size_type> rank_c_i_ = vector<size_type>(index.wavelet_tree.sigma);
    vector<size_type> rank_c_j_ = vector<size_type>(index.wavelet_tree.sigma);

    vector<size_type> ret;
    if (low == high) return ret;
    size_type quantity;                          // quantity of characters in interval
    interval_symbols(index.wavelet_tree, low, high, quantity, chars_, rank_c_i_, rank_c_j_);
    for (size_type i = 0; i < quantity; i++)
    { 
        
        ret.push_back(chars_[i]);
        ret.push_back((size_type) rank_c_j_[i] - rank_c_i_[i]);
    }
    return ret; 
}

const std::pair<size_type, size_type> FMIndex::get_range(const std::vector<size_type>& sequence)  // Finds the FM-index rows that match the input prefix `sequence`.
{
    size_type start_row = 0;
    size_type end_row = this->size();
    // Check if the sequence is empty
    if (sequence.empty()) {
        return std::make_pair(start_row, end_row);
    }
    // std::cout << "start_row: " << start_row << " end_row: " << end_row << endl;
    for (const auto& token : sequence) {
        auto results = this->backward_search_step(token + SHIFT, start_row, end_row);
        start_row = results[0];
        end_row = results[1];
        // std::cout << "start_row: " << start_row << " end_row: " << end_row << endl;
    }
    end_row += 1;
    return std::make_pair(start_row, end_row);
}

const std::vector<size_type> FMIndex::get_distinct(size_type low, size_type high)  // Finds all distinct symbols that appear in the last column of the FM-index in a given range.
{
    std::vector<size_type> distinct = this->distinct(low, high);
    std::vector<size_type> adjusted_distinct;
    for (auto c : distinct) {
        if (c > 0) {
            // std::cout << c << std::endl;
            adjusted_distinct.push_back(c - SHIFT);
        }
    }
    return adjusted_distinct;
}

const vector<vector<size_type>> FMIndex::distinct_count_multi(vector<size_type> lows, vector<size_type> highs)
{
    vector<vector<size_type>> ret;
    vector<std::future<const vector<size_type>>> threads;
    for (size_type i = 0; i < lows.size(); i++) {
        threads.push_back(
            std::async(&FMIndex::distinct_count, this, lows[i], highs[i])
        );
    }
    for (size_type i = 0; i < lows.size(); i++) {
        ret.push_back(
            threads[i].get()
        );
    }
    return ret;
}

size_type FMIndex::locate(size_type row)
{
    if (row >= index.size()) return -1;
    return (size_type) index[row];
}

const vector<size_type> FMIndex::extract_text(size_type begin, size_type end)
{
    vector<size_type> ret;
    if (end - begin == 0) return ret;
    if (end >= index.isa.size()) {
        // std::cerr << "Error: 'end' index out of isa bounds." << std::endl;
        end = index.isa.size() - 1;
    }
    size_type start = index.isa[end];
    size_type symbol = index.bwt[start];
    ret.push_back(symbol - SHIFT);
    if (end - begin == 1) return ret;
    for (size_type i = 0; i < end-begin-1; i++) 
    {
        start = backward_search_step(symbol, start, start+1)[0];
        symbol = index.bwt[start];
        ret.push_back(symbol - SHIFT);
    }
    return ret; 
}




void FMIndex::initialize_with_doc_info(const std::vector<std::pair<std::string, std::vector<size_type>>>& documents) {
    std::vector<size_type> concatenated_data;
    size_type current_position = 0;
    size_type doc_id = 0;
    for (const auto& item : documents) {
        std::string doc_id_str = item.first;
        doc_id_map[doc_id] = doc_id_str;
        const auto& doc = item.second;
        std::vector<size_type> reversedDoc;
        for (auto it = doc.rbegin(); it != doc.rend(); ++it) {
            reversedDoc.push_back(*it + SHIFT);
        }
        concatenated_data.insert(concatenated_data.end(), reversedDoc.begin(), reversedDoc.end());
        // Store document boundary
        doc_boundaries.push_back(current_position + reversedDoc.size());
        current_position += reversedDoc.size();
        doc_id++;
    }
    // Initialize the FM-index with the concatenated data
    initialize(concatenated_data);
}

std::unordered_map<std::string, size_type> FMIndex::ngram_occurrence_count(size_type low, size_type high) {
    std::unordered_map<std::string, size_type> doc_occurrences;
    for (size_type i = low; i < high; ++i) {
        size_type position = locate(i);
        size_type doc_id = get_doc_id_by_position(position);
        doc_occurrences[doc_id_map[doc_id]]++;
    }
    return doc_occurrences;
}

size_type FMIndex::get_doc_id_by_position(size_type position) const {
    // 使用二分查找找到文档边界，确定position所在的文档
    auto it = upper_bound(doc_boundaries.begin(), doc_boundaries.end(), position);
    size_type doc_id = distance(doc_boundaries.begin(), it);
    return doc_id;
}


void FMIndex::save(const string& path) const {
    ofstream out(path, ios::binary);

    // 保存索引数据
    serialize(index, out);

    // 保存doc_boundaries
    size_type boundaries_size = doc_boundaries.size();
    out.write(reinterpret_cast<const char*>(&boundaries_size), sizeof(boundaries_size));
    out.write(reinterpret_cast<const char*>(doc_boundaries.data()), boundaries_size * sizeof(size_type));

    // 保存doc_id_map
    size_type doc_id_map_size = doc_id_map.size();
    out.write(reinterpret_cast<const char*>(&doc_id_map_size), sizeof(doc_id_map_size));
    for (const auto& item : doc_id_map) {
        const auto& doc_id = item.first;
        const auto& doc_id_str = item.second;

        out.write(reinterpret_cast<const char*>(&doc_id), sizeof(doc_id));

        size_type doc_id_str_length = doc_id_str.size();
        out.write(reinterpret_cast<const char*>(&doc_id_str_length), sizeof(doc_id_str_length));
        out.write(doc_id_str.data(), doc_id_str_length);
    }
}

FMIndex load_FMIndex(const std::string path) {
    ifstream in(path, ios::binary);
    FMIndex fm;

    // 加载索引数据
    load(fm.index, in);

    // 加载doc_boundaries
    size_type boundaries_size;
    in.read(reinterpret_cast<char*>(&boundaries_size), sizeof(boundaries_size));
    fm.doc_boundaries.resize(boundaries_size);
    in.read(reinterpret_cast<char*>(fm.doc_boundaries.data()), boundaries_size * sizeof(size_type));

    // 加载doc_id_map
    size_type doc_id_map_size;
    in.read(reinterpret_cast<char*>(&doc_id_map_size), sizeof(doc_id_map_size));
    for (size_type i = 0; i < doc_id_map_size; ++i) {
        size_type doc_id;
        in.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));

        size_type doc_id_str_length;
        in.read(reinterpret_cast<char*>(&doc_id_str_length), sizeof(doc_id_str_length));

        std::string doc_id_str(doc_id_str_length, '\0');
        in.read(&doc_id_str[0], doc_id_str_length);

        fm.doc_id_map[doc_id] = doc_id_str;
    }

    fm.chars = vector<size_type>(fm.index.wavelet_tree.sigma);
    fm.rank_c_i = vector<size_type>(fm.index.wavelet_tree.sigma);
    fm.rank_c_j = vector<size_type>(fm.index.wavelet_tree.sigma);
    fm.occurring_distinct = fm.get_distinct(0, fm.size());
    return fm;
}





void FMIndexManager::addDoc(const std::string& docid, const std::vector<size_type>& data) {
    FMIndex fmIndex;
    std::vector<size_type> reversedData;
    for (auto it = data.rbegin(); it != data.rend(); ++it) {
        reversedData.push_back(*it + SHIFT);
    }
    fmIndex.initialize(reversedData);
    docid_to_fm_index[docid] = fmIndex;
}

void FMIndexManager::initialize(const std::vector<std::pair<std::string, std::vector<size_type>>>& documents) {
    for (const auto& doc : documents) {
        addDoc(doc.first, doc.second);
    }
}

void FMIndexManager::saveAll(const std::string& file_path) {
    std::cout << "saveAll..." << std::endl;
    ofstream out(file_path, ios::binary);
    size_type map_size = docid_to_fm_index.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
    for (const auto& pair : docid_to_fm_index) {
        const std::string& docid = pair.first;
        size_type docid_length = docid.length();
        out.write(reinterpret_cast<const char*>(&docid_length), sizeof(docid_length));
        out.write(docid.c_str(), docid_length);
        serialize(pair.second.index, out);
    }
    out.close();
}

void FMIndexManager::loadAll(const std::string& file_path) {
    std::cout << "loadAll..." << std::endl;
    ifstream in(file_path, ios::binary);
    size_type map_size;
    in.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
    int cnt = 0;
    for (size_type i = 0; i < map_size; i++) {
        size_type docid_length;
        in.read(reinterpret_cast<char*>(&docid_length), sizeof(docid_length));
        std::string docid(docid_length, '\0');
        in.read(&docid[0], docid_length);
        // std::cout << docid << std::endl;
        FMIndex fm;
        load(fm.index, in);
        fm.chars = vector<size_type>(fm.index.wavelet_tree.sigma);
        fm.rank_c_i = vector<size_type>(fm.index.wavelet_tree.sigma);
        fm.rank_c_j = vector<size_type>(fm.index.wavelet_tree.sigma);
        fm.occurring_distinct = fm.get_distinct(0, fm.size());
        // in.read(reinterpret_cast<char*>(&fm.occurring_distinct), sizeof(fm.occurring_distinct));
        docid_to_fm_index[docid] = fm;
        cnt += 1;
    }
    in.close();
    std::cout << "Total fm_index num:" << cnt << std::endl;
}

FMIndex& FMIndexManager::getFMIndex(const std::string& docid) {
    return docid_to_fm_index[docid];
}

size_type FMIndexManager::getDocCount() {
    return docid_to_fm_index.size();
}


/*
运行命令：
ulimit -a
ulimit -s 1024000
ulimit -c unlimited
cd /home/u2023000153/Projects/2024/GenRAG/seal/cpp_modules_new
cd /home/u2023000153/Baselines/GR-models/SEAL/seal/cpp_modules_new
g++ -DNDEBUG -std=c++11 -o fm_index  -I/home/u2023000153/include -L/home/u2023000153/lib  fm_index.cpp  -lsdsl -ldivsufsort -ldivsufsort64 -lpthread  &&  ./fm_index
*/


int main() {
    // 创建一些示例文档
    std::vector<std::pair<std::string, std::vector<size_type>>> documents = {
        {"doc1", {10, 12, 13, 14, 15}},
        {"doc2", {12, 13, 14, 15, 16}},
        {"doc3", {13, 14, 15, 16, 17, 13, 14, 99}},
        {"doc4", {13, 14, 20, 30, 40, 13, 14, 32555, 30645}}, 
    };

    // FMIndex fmindex_prev;
    
    // // 测试initialize_with_doc_info
    // std::cout << "Testing initialize_with_doc_info:\n";
    // fmindex_prev.initialize_with_doc_info(documents);

    // std::string save_path = "./corpus_fm_index_test.bin";
    // fmindex_prev.save(save_path);

    // FMIndex fmindex = load_FMIndex(save_path);

    // // 测试ngram_occurrence_count
    // std::cout << "\nTesting ngram_occurrence_count:\n";
    // // 假设我们要查找在整个索引范围内的n-gram出现次数
    // auto occurrences = fmindex.ngram_occurrence_count(0, fmindex.size());
    // for (const auto& [doc_id, count] : occurrences) {
    //     std::cout << "Document " << doc_id << ": " << count << endl;
    // }

    // std::cout << "\nTesting ngram_occurrence_count:\n";
    // auto range = fmindex.get_range({13, 14});
    // auto occurrences2 = fmindex.ngram_occurrence_count(range.first, range.second);
    // std::cout << "range: (" << range.first << ", " << range.second << ")" << std::endl;

    // for (const auto& [doc_id, count] : occurrences2) {
    //     std::cout << "Document " << doc_id << ": " << count << endl;
    // }


    // FM-Index Map
    std::string fm_index_map_path = "./fm_index_map.bin";

    // 初始化FMIndexManager
    FMIndexManager fmManager;
    fmManager.initialize(documents);
    // fmManager.initialize_multiprocess(documents);

    // 保存所有索引到目录
    fmManager.saveAll(fm_index_map_path);

    // 假设程序重启，我们需要从磁盘加载所有索引
    FMIndexManager newFmManager;
    newFmManager.loadAll(fm_index_map_path);
    std::cout << "FM-Index Loaded" << std::endl;

    // 使用加载的索引执行一些查询
    auto fmIndex = newFmManager.getFMIndex("doc4");  // 获取文档doc2的FMIndex
    std::cout << "occurring_distinct: " << fmIndex.occurring_distinct << std::endl;

    auto range = fmIndex.get_range({13, 14});  // 查询包含序列{13, 14}的范围
    std::cout << "range: (" << range.first << ", " << range.second << ")" << std::endl;

    auto distinct = fmIndex.get_distinct(range.first, range.second);  // 
    std::cout << "distinct: " << distinct << std::endl;

    return 0;
}


