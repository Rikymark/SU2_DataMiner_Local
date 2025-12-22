#include <iostream>
#include <string>
#include <vector>
#define MLP_CUSTOM_TYPE double;
#include "MLPCpp/include/CLookUp_ANN.hpp"

class MLPCppEvaluator {
    MLPToolbox::CLookUp_ANN * lookup_mlp = nullptr;
    MLPToolbox::CIOMap * iomap = nullptr;
    std::string*mlp_filenames = nullptr;
    size_t n_mlps = 0;
    std::vector<std::string> mlp_filenames_vector;
    std::vector<std::string> query_vars_in;
    std::vector<std::string> query_vars_out;
    std::vector<double> query_out;
    std::vector<double*> query_refs_out;
    public: 
    MLPCppEvaluator();
    ~MLPCppEvaluator();
    void AddMLP(std::string);
    void DisplayMLPs();
    void GenerateMLP();
    void SetQueryInputs(std::vector<std::string>);
    void SetQueryOutputs(std::vector<std::string>);
    std::vector<std::vector<double>> EvaluateMLP(std::vector<std::vector<double>>);
    double GetMLPOuput(int);
};