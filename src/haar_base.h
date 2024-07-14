#ifndef HAAR_BASE_H
#define HAAR_BASE_H
#include "link_list.h"
#include "boost_cfg.h"
#include "adaboost_base.h"
#include "WeakLearner/weaklearner.h"
/**
 * \file haar_base.h
 * \brief 基于哈尔特征的 Adaboost 分类器--基类定义及函数声明
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 采用哈尔特征的弱学习器及其系数结构体
struct haar_wl {
	flt_t alpha;			///< 弱学习器系数
	unsigned char weaklearner[];	///< 弱学习器
};

/// 采用哈尔特征的 Adaboost 强学习器
/** 注：如果 using_fold 为真，则链表 wl 中保存弱学习器的地址；
 * 如果 using_fold 为假，则链表 wl 中保存 struct haar_wl */
struct haar_adaboost {
	bool using_fold;		///< 系数 alpha 是否并入弱学习器的标志
	struct link_list wl;		///< 弱学习器链表
	flt_t threshold;		///< 分类的阈值
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief 从文件中读取 haar_adaboost
 * \details \copydetails haar_ada_read_fn
 */
bool haar_ada_read(struct haar_adaboost *adaboost, FILE * file,
		   const struct wl_handles *handles);

/**
 * \brief 将 haar_adaboost 结构体写入到文件
 * \details \copydetails haar_ada_write_fn
 */
bool haar_ada_write(const struct haar_adaboost *adaboost, FILE * file,
		    const struct wl_handles *handles);

/**
 * \brief 对 haar_adaboost 进行深度复制
 * \details \copydetails haar_ada_copy_fn
 */
void *haar_ada_copy(struct haar_adaboost *dst,
		    const struct haar_adaboost *src,
		    const struct wl_handles *handles);

/**
 * \brief haar_adaboost 结构体成员内存释放方法
 * \details \copydetails haar_ada_free_fn
 */
void haar_ada_free(struct haar_adaboost *adaboost,
		   const struct wl_handles *handles);

#endif
