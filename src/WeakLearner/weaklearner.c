#include "weaklearner.h"
#include "constant/constant.h"
#include "stump/vec_stump.h"
#include "stump/haar_stump.h"
#include "stump/haar_stump_ga.h"
/**
 * \file weaklearner.c
 * \brief 弱学习器函数调用集相关函数实现。
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    函数定义
 ******************************************************************************/
void wl_set_constant(struct wl_handles *handles)
{
	handles->size = sizeof(constant);
	handles->using_confident = false;
	handles->hypothesis.vec = constant_h;
	handles->train.vec = constant_train;
	handles->read = NULL;
	handles->write = NULL;
	handles->copy = NULL;
	handles->free = NULL;
}

void wl_set_vec_cstump(struct wl_handles *handles)
{
	handles->size = sizeof(struct vec_cstump);
	handles->using_confident = false;
	handles->hypothesis.vec = vec_cstump_h;
	handles->train.vec = vec_cstump_train;
	handles->read = vec_cstump_read;
	handles->write = vec_cstump_write;
	handles->copy = NULL;
	handles->free = NULL;
}

void wl_set_vec_cstump_cf(struct wl_handles *handles)
{
	handles->size = sizeof(struct vec_cstump_cf);
	handles->using_confident = true;
	handles->hypothesis.vec_cf = vec_cstump_cf_h;
	handles->train.vec = vec_cstump_cf_train;
	handles->read = vec_cstump_cf_read;
	handles->write = vec_cstump_cf_write;
	handles->copy = NULL;
	handles->free = NULL;
}

void wl_set_vec_dstump(struct wl_handles *handles)
{
	handles->size = sizeof(struct vec_dstump);
	handles->using_confident = false;
	handles->hypothesis.vec = vec_dstump_h;
	handles->train.vec = vec_dstump_train;
	handles->read = vec_dstump_read;
	handles->write = vec_dstump_write;
	handles->copy = vec_dstump_copy;
	handles->free = vec_dstump_free;
}

void wl_set_vec_dstump_cf(struct wl_handles *handles)
{
	handles->size = sizeof(struct vec_dstump_cf);
	handles->using_confident = true;
	handles->hypothesis.vec_cf = vec_dstump_cf_h;
	handles->train.vec = vec_dstump_cf_train;
	handles->read = vec_dstump_cf_read;
	handles->write = vec_dstump_cf_write;
	handles->copy = vec_dstump_cf_copy;
	handles->free = vec_dstump_cf_free;
}

void wl_set_haar(struct wl_handles *handles)
{
	handles->size = sizeof(struct haar_stump);
	handles->using_confident = false;
	handles->hypothesis.haar = haar_stump_h;
	handles->train.haar = haar_stump_train;
	handles->read = NULL;
	handles->write = NULL;
	handles->copy = NULL;
	handles->free = NULL;
}

// 将回调函数集设为 Haar 决策树桩，带置信度
void wl_set_haar_cf(struct wl_handles *handles)
{
	handles->size = sizeof(struct haar_stump_cf);
	handles->using_confident = true;
	handles->hypothesis.haar_cf = haar_stump_cf_h;
	handles->train.haar = haar_stump_cf_train;
	handles->read = NULL;
	handles->write = NULL;
	handles->copy = NULL;
	handles->free = NULL;
}

void wl_set_haar_ga(struct wl_handles *handles)
{
	handles->size = sizeof(struct haar_stump);
	handles->using_confident = false;
	handles->hypothesis.haar = haar_stump_h;
	handles->train.haar = haar_stump_ga_train;
	handles->read = NULL;
	handles->write = NULL;
	handles->copy = NULL;
	handles->free = NULL;
}

void wl_set_haar_ga_cf(struct wl_handles *handles)
{
	handles->size = sizeof(struct haar_stump_cf);
	handles->using_confident = true;
	handles->hypothesis.haar_cf = haar_stump_cf_h;
	handles->train.haar = haar_stump_ga_cf_train;
	handles->read = NULL;
	handles->write = NULL;
	handles->copy = NULL;
	handles->free = NULL;
}
