#pragma once

#include <sdsl/suffix_arrays.hpp>


typedef sdsl::csa_wt_int<> fm_index_type;
typedef unsigned long size_type;


class FMIndex {
public:
    FMIndex();
    ~FMIndex();
    void initialize(const std::vector<size_type> &data);
    void initialize_from_file(const std::string file, int width);
    const std::vector<size_type> backward_search_multi(const std::vector<size_type> query);
    const std::vector<size_type> backward_search_step(size_type symbol, size_type low, size_type high);
    const std::vector<size_type> distinct(size_type low, size_type high);
    const std::vector<size_type> distinct_count(size_type low, size_type high);
    const std::vector<std::vector<size_type>> distinct_count_multi(std::vector<size_type> lows, std::vector<size_type> highs);     
    size_type size();
    size_type locate(size_type row);
    const std::vector<size_type> extract_text(size_type begin, size_type end);
    void save(const std::string& path) const;
    sdsl::csa_wt_int<> index;
    std::vector<size_type> chars;
    std::vector<size_type> rank_c_i;
    std::vector<size_type> rank_c_j;
    
    const std::pair<size_type, size_type> get_range(const std::vector<size_type>& sequence);
    const std::vector<size_type> get_distinct(size_type low, size_type high);
    void initialize_with_doc_info(const std::vector<std::pair<std::string, std::vector<size_type>>>& documents);
    std::unordered_map<std::string, size_type> ngram_occurrence_count(size_type low, size_type high);
    size_type get_doc_id_by_position(size_type position) const;
    std::vector<size_type> occurring_distinct;
    std::vector<size_type> doc_boundaries;
    std::unordered_map<size_type, std::string> doc_id_map;
    // std::unordered_map<size_type, size_type> position_to_doc_id;

private:
    sdsl::int_vector<> query_;
};

FMIndex load_FMIndex(const std::string path);


class FMIndexManager {
public:
    std::unordered_map<std::string, FMIndex> docid_to_fm_index;
    void addDoc(const std::string& docid, const std::vector<size_type>& data);
    void initialize(const std::vector<std::pair<std::string, std::vector<size_type>>>& documents);
    // void initialize_multiprocess(const std::vector<std::pair<std::string, std::vector<size_type>>>& documents);
    // void processChunk(const std::vector<std::pair<std::string, std::vector<size_type>>>& chunk, const std::string& temp_file_path);
    // void initialize_and_save_multiprocess(
    //     const std::vector<std::pair<std::string, std::vector<size_type>>>& documents, 
    //     const std::string& temp_dir_path, const std::string& final_file_path
    //     );
    void saveAll(const std::string& fm_index_dir_path);
    void loadAll(const std::string& fm_index_dir_path);
    FMIndex& getFMIndex(const std::string& docid);
    size_type getDocCount();
private:
    std::mutex mtx;
    std::atomic<size_t> processed_count{0};
    std::atomic<size_t> total_count{0};
};
