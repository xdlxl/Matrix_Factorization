// Copyright (c) 2014-2015 The AsyncFTRL Project
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

#include <unistd.h>
#include <cstdlib>
#include <utility>
#include <vector>
#include "src/file_parser.h"
#include "src/mf_solver.h"
#include "src/util.h"

void print_usage(int argc, char* argv[]) {
	printf("Usage:\n");
	printf("\t%s -t test_file -m model \n", argv[0]);
}
template<typename T>
bool LoadBatchSamples(FileParser<T>& file_parser,
          std::vector<T>& train_samples_scores,
          std::vector<std::vector<int> >& train_samples,
          int batch_size){
	int cnt = 0;
	T score = 0.;
	std::vector<int> x;
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

int main(int argc, char* argv[]) {
	int ch;

	std::string test_file;
	std::string model_file;

	while ((ch = getopt(argc, argv, "t:m:h")) != -1) {
		switch (ch) {
		case 't':
			test_file = optarg;
			break;
		case 'm':
			model_file = optarg;
			break;
		case 'h':
		default:
			print_usage(argc, argv);
			exit(0);
		}
	}

	if (test_file.size() == 0 || model_file.size() == 0 ) {
		print_usage(argc, argv);
		exit(1);
	}

	MFModel<double> model;
	model.Initialize(model_file.c_str());

	int num_threads = 8;
	
	std::vector<std::string> split_train_list;
	split_trainfiles(test_file.c_str(),split_train_list,num_threads);
	if(split_train_list.size() < num_threads )
		num_threads = split_train_list.size();

	size_t cnt = 0, correct = 0;
	FileParser<double> parser;
	parser.OpenFile(test_file.c_str());


	int batch_size = 100000;
	int count = 0;
	SpinLock lock;
	double global_rmse = 0.;

	auto worker_func = [&] (size_t i) {

		FileParser<double> parser;
		parser.OpenFile(split_train_list[i].c_str());

		size_t local_count = 0;
		std::vector<std::vector<int> > train_samples;
		std::vector<double> train_samples_scores;
		while (LoadBatchSamples<double>(parser,train_samples_scores,train_samples,batch_size)){
			double local_rmse = 0.;
			double score = 0.;
			for( size_t j = 0; j < train_samples.size();j++){
				std::vector<int>& tx = train_samples[j];
				score = train_samples_scores[j];
				local_rmse += model.Predict(score,tx);
			}
			train_samples.clear();
			local_count = batch_size;
			{
					std::lock_guard<SpinLock> lockguard(lock);
					count += local_count;
					global_rmse += local_rmse;
			}
		}//while
	};

	util_parallel_run(worker_func, num_threads);

	parser.CloseFile(); 

	printf("RMSE =%lf\n", sqrt(global_rmse/ count));


	return 0;
}
/* vim: set ts=4 sw=4 tw=0 noet :*/
