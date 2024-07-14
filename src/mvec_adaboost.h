// 输入样本为向量集的多分类 Adaboost 学习器基类
#ifndef MVEC_ADABOOST_H
#define MVEC_ADABOOST_H
#include "boost_cfg.h"
#include "WeakLearner/weaklearner.h"

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
// 输入样本为向量集的多分类 Adaboost 学习器
struct mvec_adaboost {
	turn_t group_len;		// 弱学习器分组数量
	mlabel_t dim;			// 单个弱学习器分组中弱学习器的数量
	unsigned char *weaklearner;	// 弱学习器数组地址
	flt_t *alpha;			// 弱学习器系数数组的地址
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/

/*
 * 初始化 Adaboost
 * ada: 指向未初始化的结构体
 * group_len: 弱学习器分组数量
 * dim: 单个弱学习器分组中弱学习器的数量
 * using_fold: 表示是否将弱学习器系数并入弱学习器
 * handles: 弱学习器回调函数集合
 * 返回值：内存分配成功返回真，否则返回假
 */
bool mvec_ada_init(struct mvec_adaboost *ada, turn_t group_len,
		   mlabel_t dim, bool using_fold,
		   const struct wl_handles *handles);

/*
 * 从文件中读取 Adaboost
 * adaboost: 指向未初始化的 struct mvec_adaboost 结构体
 * file: 已打开的文件
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回真；失败则返回假
 */
bool mvec_ada_read(struct mvec_adaboost *adaboost, FILE * file,
		   const struct wl_handles *handles);

/*
 * 将 Adaboost 结构体写入到文件
 * adaboost: 指向已保存训练结果的 struct mvec_adaboost 结构体
 * file: 已打开的文件
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回真；失败则返回假
 */
bool mvec_ada_write(const struct mvec_adaboost *adaboost, FILE * file,
		    const struct wl_handles *handles);

/*
 * 对 Adaboost 进行深度复制
 * dst: 指向未初始化的 struct mvec_adaboost 结构体
 * src: 指向已初始化的 struct mvec_adaboost 结构体，内容将被复制到 dst
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回 dst；失败则返回 NULL
 */
void *mvec_ada_copy(struct mvec_adaboost *dst,
		    const struct mvec_adaboost *src,
		    const struct wl_handles *handles);

/*
 * 内存释放方法
 * adaboost: 指向已保存训练结果的 struct mvec_adaboost 结构体
 * handles: 弱学习器回调函数集合
 */
void mvec_ada_free(struct mvec_adaboost *adaboost,
		   const struct wl_handles *handles);

#endif
