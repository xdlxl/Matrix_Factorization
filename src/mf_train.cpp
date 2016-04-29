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

#include <getopt.h>
#include <unistd.h>
#include <iostream>
#include <locale>
#include "src/fast_mf_solver.h"
#include "src/mf_train.h"
#include "src/util.h"

using namespace std;

void print_usage() {
	printf("Usage: ./mf_train -f input_file -m model_file [options]\n"
		"options:\n"
		"--epoch iteration : set number of iteration, default 1\n"
		"--alpha alpha : set learning rate param, default 0.1\n"
		"--l2 l2 : set l2 param, default 0\n"
		"--thread num : set thread num, default is 2 threads. 0 will use hardware concurrency\n"
		"--double-precision : set to use double precision, default false\n"
		"--batch_size : set num of samples load in batch\n"
		"--help : print this help\n"
	);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{

	int len;
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

template<typename T>
bool train(const char* input_file,  const char* model_file,
		T alpha, T l2, 	size_t epoch, size_t push_step, size_t fetch_step, size_t num_threads, int batch_size) {
		FastMFTrainer<T> trainer;
		trainer.Initialize(epoch, num_threads, push_step, fetch_step);
		trainer.Train(alpha, l2, model_file, input_file);
		return true;
	}


int main(int argc, char* argv[]) {
	int opt;
	int opt_idx = 0;

	static struct option long_options[] = {
		{"epoch", required_argument, NULL, 'i'},
		{"alpha", required_argument, NULL, 'a'},
		{"l2", required_argument, NULL, 'e'},
		{"thread", required_argument, NULL, 'n'},
		{"double-precision", no_argument, NULL, 'x'},
		{"help", no_argument, NULL, 'h'},
		{"batch_size", required_argument, NULL, 'y'},
		{0, 0, 0, 0}
	};

	std::string input_file;
	std::string test_file;
	std::string model_file;
	std::string weight_file;
	std::string start_from_model;

	double alpha = DEFAULT_ALPHA;
	double l2 = DEFAULT_L2;

	size_t epoch = 1;
	bool cache = true;
	size_t push_step = kPushStep;
	size_t fetch_step = kFetchStep;
	size_t num_threads = 2;
	bool lock_free = false;
    int batch_size  = 1000000;
    float comb_prob = 0.0;

	double burn_in_phase = 0;

	bool double_precision = false;

	while ((opt = getopt_long(argc, argv, "f:m:ch", long_options, &opt_idx)) != -1) {
		switch (opt) {
		case 'f':
			input_file = optarg;
			break;
		case 'm':
			model_file = optarg;
			break;
		case 'i':
			epoch = (size_t)atoi(optarg);
			break;
		case 'a':
			alpha = atof(optarg);
			break;
		case 'e':
			l2 = atof(optarg);
			break;
		case 's':
			push_step = (size_t)atoi(optarg);
			fetch_step = push_step;
			break;
		case 'n':
			num_threads = (size_t)atoi(optarg);
			break;
		case 'x':
			double_precision = true;
			break;
		case 'y':
			batch_size = atoi(optarg);
			break;
		case 'h':
		default:
			print_usage();
			exit(0);
		}
	}

	if (input_file.size() == 0 || model_file.size() == 0) {
		print_usage();
		exit(1);
	}


	if (double_precision) {
		train<double>(input_file.c_str(),  model_file.c_str(),alpha, l2, 
			epoch, push_step, fetch_step, num_threads, batch_size);
	} else {
		train<float>(input_file.c_str(),  model_file.c_str(),alpha, l2, 
			epoch, push_step, fetch_step, num_threads, batch_size);
	}

	return 0;
}
/* vim: set ts=4 sw=4 tw=0 noet :*/
