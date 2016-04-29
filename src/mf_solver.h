// Copyright (c) 2014-2015 The AsyncMF Project
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_MF_SOLVER_H
#define SRC_MF_SOLVER_H

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <strstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include "src/util.h"

#define DEFAULT_ALPHA 0.01
#define DEFAULT_BETA 1.
#define DEFAULT_L1 1.
#define DEFAULT_L2 1.
const double rand_val = 0.1;
int dim = 20;

template<typename T>
class MFSolver {
	public:
	MFSolver();

	virtual ~MFSolver();

	virtual bool Initialize(
		T alpha,T l2, size_t user_num,size_t item_num,int latent_dim);

	virtual bool Initialize(const char* path);

	virtual bool SaveModelAll(const char* path);
	virtual bool SaveModel(const char* path);
	virtual bool SaveModelDetail(const char* path);

	public:
	T alpha() { return alpha_; }
	T l2() { return l2_; }
	int  l_dim() { return l_dim_; }
	size_t feat_num() { return feat_num_; }
	size_t user_num() { return user_num_; }

	protected:
	enum {kPrecision = 8};

	protected:
	T GetWeight(size_t row,size_t col);
	T GetWeightSave(size_t row,size_t col);
    void set_float_rand(T** x, size_t n,const int l_dim, T val);

	protected:
	T alpha_;
	T l2_;
	size_t feat_num_;
	size_t user_num_;
	size_t item_num_;
    int l_dim_;

    //matrix factorization 
    T ** u_; //user + title latent matrix //the fisrt max_user_key is user latent factor matrix
    //T ** v_; //title word latent matrix

	bool init_;

	std::mt19937 rand_generator_;
	std::uniform_real_distribution<T> uniform_dist_;
};



template<typename T>
MFSolver<T>::MFSolver()
: alpha_(0), l2_(0), feat_num_(0),u_(NULL),init_(false),user_num_(0),item_num_(0),
uniform_dist_(0.0, std::nextafter(1.0, std::numeric_limits<T>::max())) {}

template<typename T>
MFSolver<T>::~MFSolver() {
    if (u_){
	    for (size_t i = 0; i < feat_num_; ++i) 
        {
            if (u_[i] )
                delete [] u_[i];
        }
        delete [] u_;
    }
}

template<typename T>
void set_float_zero(T* x, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		x[i] = 0;
	}
}

template<typename T>
void MFSolver<T>::set_float_rand(T** x, size_t n,const int l_dim_, T val){
    if (val == 0.0)
     {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < l_dim_; ++j) {
                x[i][j] = 0.0;
            }
        }
         return;
     }
     std::random_device rd;
     std::mt19937 gen(rd());
     std::default_random_engine generator;
     //std::normal_distribution<T> distribution(0,rand_val);

	 float scale = sqrt(1.0/l_dim_);//lib_mf method
     std::uniform_real_distribution<> distribution(0,1.);
	for (size_t i = 0; i < n; ++i) {
	    for (size_t j = 0; j < l_dim_; ++j) {
            x[i][j] = distribution(generator) * scale;
	    }
    }
}

template<typename T>
bool MFSolver<T>::Initialize(
		T alpha,
		T l2,
		size_t user_num,size_t item_num,int latent_dim) {
	alpha_ = alpha;
	l2_ = l2;
	user_num_ = user_num;
	item_num_ = item_num;
	feat_num_ = user_num + item_num;//using one large matrix store user and item latent factors
    l_dim_ = latent_dim;
	u_ = new T*[feat_num_];
    for (size_t i = 0; i < feat_num_; ++i) 
        u_[i] = new T[l_dim_];
    set_float_rand(u_,feat_num_,l_dim_,0.01);
	init_ = true;
	return init_;
}

template<typename T>
bool MFSolver<T>::Initialize(const char* path) {
	std::fstream fin;
	fin.open(path, std::ios::in);
	if (!fin.is_open()) {
		return false;
	}

	init_ = true;
	return init_;
}

template<typename T>
T MFSolver<T>::GetWeight(size_t row,size_t col) {
	return u_[row][col];
}
template<typename T>
T MFSolver<T>::GetWeightSave(size_t row,size_t col) {
        return u_[row][col];
}

template<typename T>
bool MFSolver<T>::SaveModel(const char* path) {
	if (!init_) return false;

	std::fstream fout;
	std::ios_base::sync_with_stdio(false);
	fout.open(path, std::ios::out);

	if (!fout.is_open()) {
		return false;
	}

	fout << std::fixed << std::setprecision(kPrecision);
    //save feature dimension
    fout << user_num_ << "\n";
    fout << item_num_ << "\n";
    //save latent factor dimension
    fout << l_dim_ << "\n";
	for (size_t i = 0; i < feat_num_; ++i) {
        for (size_t j = 0; j < l_dim_ -1; ++j) {
                fout << GetWeightSave(i,j)  << "\t";
        }
		fout << GetWeightSave(i,l_dim_-1)  << "\n";
	}
	fout.close();
	return true;
}

template<typename T>
bool MFSolver<T>::SaveModelDetail(const char* path) {
	if (!init_) return false;
	return true;
}

template<typename T>
bool MFSolver<T>::SaveModelAll(const char* path) {
	return SaveModel(path) ;
}



template<typename T>
class MFModel {
public:
	MFModel();
	virtual ~MFModel();

	bool Initialize(const char* path);

	T Predict(T& score,const std::vector<int>& x);
private:
	std::vector<T> model_;
	T**  v_;
    size_t feat_num_;
    size_t user_num_;
    size_t item_num_;
    int l_dim_;
	bool init_;
};

template<typename T>
MFModel<T>::MFModel() : v_(NULL),init_(false) {
    feat_num_ = 0;
    l_dim_ = 0;
}

template<typename T>
MFModel<T>::~MFModel() {

    if (v_){
	    for (size_t i = 0; i < feat_num_; ++i) 
            if (v_[i] )
                delete [] v_[i];
        delete [] v_;
    }
}

template<typename T>
bool MFModel<T>::Initialize(const char* path) {
	std::fstream fin;
	fin.open(path, std::ios::in);
	if (!fin.is_open()) {
		return false;
	}

    fin >> user_num_; 
    fin >> item_num_; 
	feat_num_ = user_num_ + item_num_;
    fin >> l_dim_ ; //get latentfactor dimension
    v_ = new T*[feat_num_];
	for (size_t i = 0; i < feat_num_; ++i) {
        v_[i] = new T[l_dim_];
	    for (size_t j = 0; j < l_dim_; ++j) {
            fin >> v_[i][j];
        }
        if (!fin || fin.eof()) {
            fin.close();
            return false;
        }
	}

	fin.close();

	init_ = true;
	return init_;
}

template<typename T>
T MFModel<T>::Predict(T& score,const std::vector<int>& x) {
	if (!init_) {
		printf("model init failed !\n");
		return 0;
	}
	T avg_rmse = 0.;
	if (x.size() < 2)
		return avg_rmse;
	int userid = x[0];

	//group each user 's same score items in each line
	for (int i = 1;i < x.size();i++) {
		int item_id = x[i] + user_num_;
		if (item_id >= feat_num_) break;
		T pred_score = 0.;
		for (size_t j=0; j < l_dim_; ++j) 
			pred_score += v_[userid][j] * v_[item_id][j];
		avg_rmse += pow(pred_score - score,2);
	}
	return avg_rmse / (x.size() - 1);
}
void split_trainfiles(const char* train_files_list,std::vector<std::string>& split_train_list,int num_threads){
		std::ifstream fin;
		fin.open(train_files_list);
		std::vector<std::string> train_files_vec;
		std::string line;
		while(getline(fin,line)){
			//printf("file %s \n",line.c_str());
			train_files_vec.push_back(line);
		}
		fin.close();
		if (train_files_vec.size() >= num_threads){
			int split_num = num_threads;
			int each_split_num = train_files_vec.size()/split_num;
			for(int i = 0; i < split_num; i++){
				std::strstream ss;	std::string istr;
				ss << i;	ss >> istr;
				int j;
				std::ofstream ofs;
				std::string ofiles = std::string(train_files_list) + "." + istr;
				ofs.open(ofiles.c_str());
				for(j = i*each_split_num; j < (i+1)*each_split_num; j++)
					ofs << train_files_vec[j] << "\n";
				if(i == split_num -1 && j < train_files_vec.size())
					for(; j < train_files_vec.size(); j++)
						ofs << train_files_vec[j] << "\n";
				ofs.close();
				split_train_list.push_back(ofiles);
			}
		}
		else{
			int split_num = train_files_vec.size();
			for(int i = 0; i < split_num; i++){
				std::strstream ss;	std::string istr;
				ss << i;	ss >> istr;
				std::ofstream ofs;
				std::string ofiles = std::string(train_files_list) + "." + istr;
				ofs.open(ofiles.c_str());
				ofs << train_files_vec[i] << "\n";
				ofs.close();
				split_train_list.push_back(ofiles);
			}
			printf("file num less than threads, files num is %d\n",split_num);
		}
}

#endif // SRC_MF_SOLVER_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
