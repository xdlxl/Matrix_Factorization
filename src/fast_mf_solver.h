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

#ifndef SRC_FAST_MF_SOLVER_H
#define SRC_FAST_MF_SOLVER_H

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <map>
#include "src/mf_solver.h"
#include "src/lock.h"

extern const double rand_val ;
enum { kParamGroupSize = 1, kFetchStep = 3, kPushStep = 3 };

inline size_t calc_group_num(size_t n) {
	return (n + kParamGroupSize - 1) / kParamGroupSize;
}

template<typename T>
class MFParamServer : public MFSolver<T> {
public:
	MFParamServer();

	virtual ~MFParamServer();

	virtual bool Initialize(
		T alpha,
		T l2,
		size_t user_num,size_t item_num,int latent_dim);
	virtual bool Initialize(const char* path);
	bool FetchParamGroup(T** u, size_t group);
	bool FetchParam(T** u);
	bool PushParamGroup(T** u, size_t group);

private:
	size_t param_group_num_;
	SpinLock* lock_slots_;
};

template<typename T>
class MFWorker : public MFSolver<T> {
public:
	MFWorker();

	virtual ~MFWorker();

	bool Initialize(
		MFParamServer<T>* param_server,
		size_t push_step = kPushStep,
		size_t fetch_step = kFetchStep);

	bool Reset(MFParamServer<T>* param_server);

	bool Initialize(
		T alpha,
		T l2,
		size_t n) { return false; }

	bool Initialize(const char* path) { return false; }

	T Update(const std::vector<int>& x,MFParamServer<T>* param_server);
	T Update(T& score,const std::vector<int>& x,MFParamServer<T>* param_server);

	bool PushParam(MFParamServer<T>* param_server);

private:
	size_t param_group_num_;
	size_t* param_group_step_;
	size_t push_step_;
	size_t fetch_step_;

	T** u_update_;
};



template<typename T>
MFParamServer<T>::MFParamServer()
: MFSolver<T>(), param_group_num_(0), lock_slots_(NULL) {}

template<typename T>
MFParamServer<T>::~MFParamServer() {
	if (lock_slots_) {
		delete [] lock_slots_;
	}
}

template<typename T>
bool MFParamServer<T>::Initialize(
		T alpha,
		T l2,
		size_t user_num,size_t item_num,int latent_dim) {
	if (!MFSolver<T>::Initialize(alpha, l2, user_num,item_num,latent_dim)) {
		return false;
	}

	size_t n = user_num + item_num;
	param_group_num_ = calc_group_num(n);
	lock_slots_ = new SpinLock[param_group_num_];

	MFSolver<T>::init_ = true;
	return true;
}

template<typename T>
bool MFParamServer<T>::Initialize(const char* path) {
	if (!MFSolver<T>::Initialize(path)) {
		return false;
	}

	param_group_num_ = calc_group_num(MFSolver<T>::feat_num_);
	lock_slots_ = new SpinLock[param_group_num_];

	MFSolver<T>::init_ = true;
	return true;
}

template<typename T>
bool MFParamServer<T>::FetchParamGroup(T** u, size_t group) {
	if (!MFSolver<T>::init_) return false;

	size_t start = group * kParamGroupSize;
	size_t end = std::min((group + 1) * kParamGroupSize, MFSolver<T>::feat_num_);

	std::lock_guard<SpinLock> lock(lock_slots_[group]);
	for (size_t i = start; i < end; ++i) {
		if (u != NULL){
			for  (int j = 0; j < MFSolver<T>::l_dim_; ++j) 
					u[i][j] = MFSolver<T>::u_[i][j];
		}
	}
	return true;
}

template<typename T>
bool MFParamServer<T>::FetchParam(T** u) {
	if (!MFSolver<T>::init_) return false;

	for (size_t i = 0; i < param_group_num_; ++i) {
		FetchParamGroup(u,i);
	}
	return true;
}

template<typename T>
bool MFParamServer<T>::PushParamGroup(T** u_update,size_t group) {
	if (!MFSolver<T>::init_) return false;

	size_t start = group * kParamGroupSize;
	size_t end = std::min((group + 1) * kParamGroupSize, MFSolver<T>::feat_num_);

	std::lock_guard<SpinLock> lock(lock_slots_[group]);
	for (size_t i = start; i < end; ++i) {
        for  (int j = 0; j < MFSolver<T>::l_dim_; ++j) { 
            MFSolver<T>::u_[i][j] += u_update[i][j];
            u_update[i][j] = 0.; 
        }
	}
	return true;
}


template<typename T>
MFWorker<T>::MFWorker()
: MFSolver<T>(), param_group_num_(0), param_group_step_(NULL),
push_step_(0), fetch_step_(0), u_update_(NULL) {}

template<typename T>
MFWorker<T>::~MFWorker() {
	if (param_group_step_) {
		delete [] param_group_step_;
	}

	if (u_update_) {
	    for (size_t i = 0; i < MFSolver<T>::feat_num_; ++i) 
            if (u_update_[i])
                delete [] u_update_[i];
        delete [] u_update_;
    }
}

template<typename T>
bool MFWorker<T>::Initialize(
		MFParamServer<T>* param_server,
		size_t push_step,
		size_t fetch_step) {
    //multi worker shared one param_server,passed by pointer MFParamServer<T>* param_server
	MFSolver<T>::alpha_ = param_server->alpha();
	MFSolver<T>::l2_ = param_server->l2();
	MFSolver<T>::feat_num_ = param_server->feat_num();
	MFSolver<T>::user_num_ = param_server->user_num();
	MFSolver<T>::l_dim_ = param_server->l_dim();

	u_update_ = new T*[MFSolver<T>::feat_num_];
    for (size_t i = 0; i < MFSolver<T>::feat_num_; ++i){
        u_update_[i] = new T[MFSolver<T>::l_dim_];
    }

    MFSolver<T>::set_float_rand(u_update_,MFSolver<T>::feat_num_,MFSolver<T>::l_dim_,0.0);

    printf("MF fea num:%ld\n",MFSolver<T>::feat_num_);
    printf("%d dim \n",MFSolver<T>::l_dim_);

	MFSolver<T>::u_ = new T*[MFSolver<T>::feat_num_];
    for (size_t i = 0; i < MFSolver<T>::feat_num_; ++i) 
        MFSolver<T>::u_[i] = new T [MFSolver<T>::l_dim_];

    MFSolver<T>::set_float_rand(MFSolver<T>::u_,MFSolver<T>::feat_num_,MFSolver<T>::l_dim_,1.);

	param_server->FetchParam(MFSolver<T>::u_);

	param_group_num_ = calc_group_num(MFSolver<T>::feat_num_);
	param_group_step_ = new size_t[param_group_num_];
	for (size_t i = 0; i < param_group_num_; ++i) param_group_step_[i] = 0;
    printf("group fea num:%ld\n",param_group_num_);

	push_step_ = push_step;
	fetch_step_ = fetch_step;

	MFSolver<T>::init_ = true;
	return MFSolver<T>::init_;
}

template<typename T>
bool MFWorker<T>::Reset(MFParamServer<T>* param_server) {
	if (!MFSolver<T>::init_) return 0;

	param_server->FetchParam(MFSolver<T>::u_);

	for (size_t i = 0; i < param_group_num_; ++i) {
		param_group_step_[i] = 0;
	}
	return true;
}


template<typename T>  
T MFWorker<T>::Update(T& score,const std::vector<int>& x,MFParamServer<T>* param_server){
		if (x.size() < 2) // must contain userid and at least one item id
		{
			printf("size less than 2\n");
			return 0.;
		}
		int user_key = x[0];
		size_t g_group = user_key / kParamGroupSize;
		if (user_key >= MFSolver<T>::user_num_) return 0.;

		float rmse = 0.;
        for( int j = 1;j < x.size();j++) {
            size_t i = x[j] + MFSolver<T>::user_num_;
			if (i >= MFSolver<T>::feat_num_) break;
            size_t g = i / kParamGroupSize;
            if (param_group_step_[g] % fetch_step_ == 0) 
                param_server->FetchParamGroup(MFSolver<T>::u_,g);
			if (param_group_step_[g_group] % fetch_step_ == 0) 
				param_server->FetchParamGroup(MFSolver<T>::u_,g_group);
			float ruv = 0.;
			for(int l = 0; l < MFSolver<T>::l_dim_;l++)
				ruv += MFSolver<T>::u_[user_key][l] * MFSolver<T>::u_[i][l];
			float obj_grad = ruv - score;
			rmse += obj_grad * obj_grad;
			for(int l = 0; l < MFSolver<T>::l_dim_;l++){
				u_update_[user_key][l] -= MFSolver<T>::alpha_ * (obj_grad * MFSolver<T>::u_[i][l]  + MFSolver<T>::l2_ * MFSolver<T>::u_[user_key][l]);
				u_update_[i][l] -= MFSolver<T>::alpha_ * (obj_grad * MFSolver<T>::u_[user_key][l] + MFSolver<T>::l2_ * MFSolver<T>::u_[i][l]);
			}

			//update
			if (param_group_step_[g_group] % push_step_ == 0)
				param_server->PushParamGroup(u_update_,g_group);
			if (param_group_step_[g] % push_step_ == 0) 
				param_server->PushParamGroup(u_update_,g);
			param_group_step_[g_group] += 1;	
			param_group_step_[g] += 1;	
    	}
		return rmse / (x.size() - 1);
}


template<typename T>
bool MFWorker<T>::PushParam(MFParamServer<T>* param_server) {
	if (!MFSolver<T>::init_) return false;

	for (size_t i = 0; i < param_group_num_; ++i) {
		param_server->PushParamGroup(u_update_,i);
	}

	return true;
}


#endif // SRC_FAST_MF_SOLVER_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
