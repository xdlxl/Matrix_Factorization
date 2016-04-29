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

#ifndef SRC_MF_TRAIN_H
#define SRC_MF_TRAIN_H

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <strstream>
#include <map>
#include "src/fast_mf_solver.h"
#include "src/file_parser.h"
#include "src/mf_solver.h"
#include "src/stopwatch.h"

const int DEFAULT_BATCH_SIZE = 100000;


template<typename T>
class FastMFTrainer {
public:
	FastMFTrainer();

	virtual ~FastMFTrainer();

	bool Initialize(
		size_t epoch,
		size_t num_threads = 0,
		size_t push_step = kPushStep,
		size_t fetch_step = kFetchStep);

	bool Train(
		T alpha,
		T l2,
		const char* model_file,
		const char* train_file);

protected:
	bool TrainImpl(
		const char* model_file,
		const char* train_file);

    bool LoadBatchSamples(FileParser<T>& file_parser,
          std::vector<T>& train_samples_scores,
          std::vector<std::vector<int> >& train_samples,
          int batch_size);
    void get_feat_num();
	//if split_train_list size less than num_threads,change num_threads_ to files number
	//void split_trainfiles(const char* train_files_list,std::vector<std::string>& split_train_list,int num_threads);
private:
	size_t epoch_;
	size_t push_step_;
	size_t fetch_step_;
	size_t user_num_;
	size_t item_num_;
	int latent_dim_;

	MFParamServer<T> param_server_;
	size_t num_threads_;

	bool init_;
};
template<typename T>                                                                                                                                  
void FastMFTrainer<T>::get_feat_num() {                                                                                                           
    std::fstream fin;                                                                                                                                 
    fin.open("./feat_num", std::ios::in);                                                                                                             
    if (!fin.is_open()) {                                                                                                                             
        printf("please supply a file name feat_num including user num ,item num and latent fatctor dimension,one each line");
		exit(-1);
    }                                                                                                                                                 
    fin >> user_num_;                                                                                                                                  
    fin >> item_num_;                                                                                                                                  
    fin >> latent_dim_;                                                                                                                                  
    fin.close();                                                                                                                                      
}      



template<typename T>
bool FastMFTrainer<T>::LoadBatchSamples(FileParser<T>& file_parser,
          std::vector<T>& train_samples_scores,
          std::vector<std::vector<int> >& train_samples,
          int batch_size){
	int cnt = 0;
	std::vector<int> x;
	T score;
	while (file_parser.ReadSample(score,x)) {
		train_samples.push_back(x);
		train_samples_scores.push_back(score);
        ++cnt;
		if (cnt >= batch_size)
			break;
    }
    if (cnt > 0 )
        return true;
    return false;
}


template<typename T>
bool FastMFTrainer<T>::Train(
		T alpha,
		T l2,
		const char* model_file,
		const char* train_file) {
	if (!init_) return false;

    get_feat_num();
	if (user_num_ == 0 || item_num_ == 0 || latent_dim_ == 0) return false;

	if (!param_server_.Initialize(alpha, l2,user_num_,item_num_,latent_dim_)){
		return false;
	}
	return TrainImpl(model_file, train_file);
}


template<typename T>
bool FastMFTrainer<T>::TrainImpl(
		const char* model_file,
		const char* train_file) {
	if (!init_) return false;

	fprintf(
		stdout,
		"params={alpha:%.4f, l2:%.4f, epoch:%zu}\n",
		static_cast<float>(param_server_.alpha()),
		static_cast<float>(param_server_.l2()),
		epoch_);

	std::vector<std::string> split_train_list;

	split_trainfiles(train_file,split_train_list,num_threads_);
	if(split_train_list.size() < num_threads_ )
		num_threads_ = split_train_list.size();

	MFWorker<T>* solvers = new MFWorker<T>[num_threads_];
	for (size_t i = 0; i < num_threads_; ++i) {
		solvers[i].Initialize(&param_server_, push_step_, fetch_step_);
	}

	StopWatch timer;
	for (size_t iter = 0; iter < epoch_; ++iter) {

		size_t cur_cnt = 0, cur_w_indx = 0;
		long long count = 0;
		double rmse = 0.;

		SpinLock lock;
		auto worker_func = [&] (size_t i) {
			FileParser<T> file_parser;
			file_parser.OpenFile(split_train_list[i].c_str());

			int	batch_size = DEFAULT_BATCH_SIZE;
			size_t local_count = 0;

			std::vector<std::vector<int> > train_samples;
			std::vector<T> train_samples_scores;

			while (LoadBatchSamples(file_parser,train_samples_scores,train_samples,batch_size) ) {
				double local_mse = 0.;
				for(int j = 0;j < train_samples.size();j++)
					local_mse += solvers[i].Update(train_samples_scores[j],train_samples[j],&param_server_);

				local_count = batch_size;
				{
					std::lock_guard<SpinLock> lockguard(lock);
					count += local_count;
					rmse += local_mse;
					if (count % DEFAULT_BATCH_SIZE == 0){
						fprintf(stdout,"epoch=%zu processed=[%lld],avg rmse is [%f] \r",iter,count,sqrt(rmse / count) );
						fflush(stdout);
					}
				}
				train_samples.clear(); 
				train_samples_scores.clear(); 
			}
        solvers[i].PushParam(&param_server_);
		file_parser.CloseFile();

	};
		for (size_t i = 0; i < num_threads_; ++i) {
			solvers[i].Reset(&param_server_);
		}

		util_parallel_run(worker_func, num_threads_);
	}

	delete [] solvers;
	return param_server_.SaveModelAll(model_file);
}
template<typename T>
FastMFTrainer<T>::FastMFTrainer()
: epoch_(0), push_step_(0),
fetch_step_(0), param_server_(), num_threads_(0), init_(false),user_num_(0),item_num_(0) { }

template<typename T>
FastMFTrainer<T>::~FastMFTrainer() {
}
template<typename T>
	bool FastMFTrainer<T>::Initialize(
		size_t epoch,
		size_t num_threads,
		size_t push_step,
		size_t fetch_step){
	
	epoch_ = epoch;
	push_step_ = push_step;
	fetch_step_ = fetch_step;
	if (num_threads == 0) {
		num_threads_ = std::thread::hardware_concurrency();
	} else {
		num_threads_ = num_threads;
	}
	init_ = true;
	return init_;
}

#endif // SRC_MF_TRAIN_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
