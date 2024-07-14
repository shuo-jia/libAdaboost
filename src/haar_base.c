#include <stdlib.h>
#include <string.h>
#include "haar_base.h"
#include "link_list.h"
/**
 * \file haar_base.c
 * \brief 基于哈尔特征的 Adaboost 分类器--函数实现
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/**
 * \brief 向文件写入或从文件读取 Adaboost
 * \param[in, out] ada    读取时，是未初始化的 Adaboost；写入时，需已初始化
 * \param[in, out] file   已打开的文件
 * \param[in] hl          弱学习器回调函数集合，const struct wl_handles * 类型
 * \param[in] frw_fun     fread 或 fwrite
 * \param[in] wl_rw       wl_read 或 wl_write
 * \param[in] link_rw_fun link_list_read 或 link_list_write
 * \return 成功返回真；否则返回假
 */
#define HAAR_RW(ada, file, hl, frw_fun, wl_rw, link_rw_fun)			\
({										\
	bool finished = false;							\
	do {									\
		if (frw_fun (&(ada)->using_fold, sizeof(bool), 1, file) < 1)	\
			break;							\
		if (frw_fun (&(ada)->threshold, sizeof(flt_t), 1, file) < 1)	\
			break;							\
		if (! link_rw_fun(&(ada)->wl, file, wl_rw,			\
					(int)(ada)->using_fold,	hl))		\
				break;						\
		finished = true;						\
	} while(0);								\
	finished;								\
})

/**
 * \brief 获取 struct haar_wl 结构体成员的地址
 * \param[in] data: void * 类型，实际上为 struct haar_wl * 类型
 * \param[in] memb: struct haar_wl 的任一成员的名称
 * \return 返回成员地址
 */
#define HAAR_PTR(data, memb) (&((struct haar_wl *)data)->memb)

/*******************************************************************************
 * 				  静态函数声明
 ******************************************************************************/
/**
 * \brief 从文件读取弱学习器
 * \param[in, out] ap: 可变参数列表，包含 int 参数（0或1，1 表示不使用弱学习器
 *      系数）以及 const struct wl_handles *（弱学习器回调函数）
 * \param[in] file:    保存弱学习器参数的文件
 * \return 成功则返回新创建的弱学习器地址，如使用弱学习器参数，则类型为
 *      struct haar_wl *，其内包含弱学习器系数；否则仅返回弱学习器地址
 */
void *wl_read(va_list ap, FILE * file);

/**
 * \brief 向文件写入弱学习器参数
 * \param[in] data      弱学习器地址
 * \param[out] ap, file 同 wl_read
 * \return 成功返回真，否则返回假
 */
bool wl_write(const void *data, va_list ap, FILE * file);

/**
 * \brief 复制弱学习器
 * \param[out] data 弱学习器地址
 * \param[in] ap    同 wl_read
 * \return 成功则返回弱学习器 data 的一份拷贝，否则返回 NULL
 */
void *wl_copy(const void *data, va_list ap);

/**
 * \brief 释放弱学习器内部使用的内存，但不包括 data 本身
 * \param[in] data: 弱学习器地址
 * \param[in] ap: 可变参数列表，包含 const struct wl_handles * hl 指针，
 *      hl->copy() 非空
 */
void wl_free(void *data, va_list ap);

/*******************************************************************************
 * 				    函数实现
 ******************************************************************************/
bool haar_ada_read(struct haar_adaboost *adaboost, FILE * file,
		   const struct wl_handles *handles)
{
	link_list_init(&adaboost->wl);
	if (!HAAR_RW(adaboost, file, handles, fread, wl_read, link_list_read)) {
		haar_ada_free(adaboost, handles);
		return false;
	}
	return true;
}

bool haar_ada_write(const struct haar_adaboost *adaboost, FILE * file,
		    const struct wl_handles *handles)
{
	return HAAR_RW(adaboost, file, handles, fwrite, wl_write,
		       link_list_write);
}

void *haar_ada_copy(struct haar_adaboost *dst,
		    const struct haar_adaboost *src,
		    const struct wl_handles *handles)
{
	dst->using_fold = src->using_fold;
	dst->threshold = src->threshold;
	link_list_init(&dst->wl);
	if (!link_list_copy_full(&dst->wl, &src->wl, wl_copy,
				 (int)dst->using_fold, handles)) {
		haar_ada_free(dst, handles);
		return NULL;
	}
	return dst;
}

void haar_ada_free(struct haar_adaboost *adaboost,
		   const struct wl_handles *handles)
{
	if (handles->free != NULL) {
		if (adaboost->using_fold)
			link_list_traverse(&adaboost->wl, handles->free);
		else
			link_list_traverse_r(&adaboost->wl, wl_free, handles);
	}
	link_list_free_full(&adaboost->wl, free);
}

/*******************************************************************************
 * 				  静态函数定义
 ******************************************************************************/
void *wl_read(va_list ap, FILE * file)
{
	int using_fold = va_arg(ap, int);
	const struct wl_handles *hl = va_arg(ap, const struct wl_handles *);
	size_t size = hl->size;
	if (!using_fold)
		size += sizeof(struct haar_wl);
	void *data = malloc(size);
	if (data == NULL)
		return NULL;

	void *wl = data;
	if (!using_fold) {	// 读取 alpha 系数
		wl = HAAR_PTR(data, weaklearner);
		if (fread(HAAR_PTR(data, alpha), sizeof(flt_t), 1, file) < 1)
			goto err;
	}
	if (hl->read == NULL) {	// 读取弱学习器
		if (fread(wl, hl->size, 1, file) < 1)
			goto err;
	} else if (!hl->read(wl, file))
		goto err;

	return data;
err:
	free(data);
	return NULL;
}

bool wl_write(const void *data, va_list ap, FILE * file)
{
	int using_fold = va_arg(ap, int);
	const struct wl_handles *hl = va_arg(ap, const struct wl_handles *);

	const void *wl = data;
	if (!using_fold) {	// 写入 alpha 系数
		wl = HAAR_PTR(data, weaklearner);
		if (fwrite(HAAR_PTR(data, alpha), sizeof(flt_t), 1, file) < 1)
			return false;
	}
	if (hl->write == NULL) {	// 写入弱学习器
		if (fwrite(wl, hl->size, 1, file) < 1)
			return false;
	} else if (!hl->write(wl, file))
		return false;

	return true;
}

void *wl_copy(const void *data, va_list ap)
{
	int using_fold = va_arg(ap, int);
	const struct wl_handles *hl = va_arg(ap, const struct wl_handles *);
	size_t size = hl->size;
	if (!using_fold)
		size += sizeof(struct haar_wl);
	void *dst = malloc(size);
	if (dst == NULL)
		return NULL;

	void *wl_dst = dst;
	const void *wl_src = data;
	if (!using_fold) {	// 复制 alpha 系数
		wl_dst = HAAR_PTR(dst, weaklearner);
		wl_src = HAAR_PTR(data, weaklearner);
		*HAAR_PTR(dst, alpha) = *HAAR_PTR(data, alpha);
	}
	if (hl->copy == NULL)	// 复制弱学习器
		memcpy(wl_dst, wl_src, hl->size);
	else if (!hl->copy(wl_dst, wl_src)) {
		free(dst);
		return NULL;
	}

	return dst;
}

void wl_free(void *data, va_list ap)
{
	const struct wl_handles *hl = va_arg(ap, const struct wl_handles *);
	hl->free(HAAR_PTR(data, weaklearner));
}
