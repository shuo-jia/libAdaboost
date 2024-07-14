// 基于哈尔特征的 Adaboost 分类器--基类
#ifndef HAAR_BASE_H
#define HAAR_BASE_H
#include "link_list.h"
#include "boost_cfg.h"
#include "adaboost_base.h"
#include "WeakLearner/weaklearner.h"

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
// 采用哈尔特征的弱学习器及其系数结构体
struct haar_wl {
	flt_t alpha;			// 弱学习器系数
	unsigned char weaklearner[];	// 弱学习器
};

// 采用哈尔特征的 Adaboost 强学习器
// 注：如果 using_fold 为真，则链表 wl 中保存弱学习器的地址；
//     如果 using_fold 为假，则链表 wl 中保存 struct haar_wl *
struct haar_adaboost {
	bool using_fold;		// 系数 alpha 是否并入弱学习器的标志
	struct link_list wl;		// 弱学习器链表
	flt_t threshold;		// 分类的阈值
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/*
 * 从文件中读取 Adaboost
 * adaboost: 指向未初始化的 struct haar_adaboost 结构体
 * file: 已打开的文件
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回真；失败则返回假
 */
bool haar_ada_read(struct haar_adaboost *adaboost, FILE * file,
		   const struct wl_handles *handles);

/*
 * 将 Adaboost 结构体写入到文件
 * adaboost: 指向已保存训练结果的 struct haar_adaboost 结构体
 * file: 已打开的文件
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回真；失败则返回假
 */
bool haar_ada_write(const struct haar_adaboost *adaboost, FILE * file,
		    const struct wl_handles *handles);

/*
 * 对 Adaboost 进行深度复制
 * dst: 指向未初始化的 struct haar_adaboost 结构体
 * src: 指向已初始化的 struct haar_adaboost 结构体，内容将被复制到 dst
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回 dst；失败则返回 NULL
 */
void *haar_ada_copy(struct haar_adaboost *dst,
		    const struct haar_adaboost *src,
		    const struct wl_handles *handles);

/*
 * 内存释放方法
 * adaboost: 指向已保存训练结果的 struct haar_adaboost 结构体
 * handles: 弱学习器回调函数集合
 */
void haar_ada_free(struct haar_adaboost *adaboost,
		   const struct wl_handles *handles);

#endif
