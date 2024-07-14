// 输入样本为向量集的多分类 Adaboost 学习器基类
#ifndef MVEC_ADABOOST_H
#define MVEC_ADABOOST_H
#include "boost_cfg.h"
#include "WeakLearner/weaklearner.h"
/**
 * \file mvec_adaboost.h
 * \brief Adaboost 分类器定义与函数声明：多分类任务、样本集为向量集
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 输入样本为向量集的多分类 Adaboost 学习器
struct mvec_adaboost {
	turn_t group_len;		///< 弱学习器分组数量
	mlabel_t dim;			///< 单个弱学习器分组中弱学习器的数量
	unsigned char *weaklearner;	///< 弱学习器数组地址
	flt_t *alpha;			///< 弱学习器系数数组的地址
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/

/**
 * \brief 初始化 mvec_adaboost
 * \param[out] ada       指向未初始化的结构体
 * \param[in] group_len  弱学习器分组数量
 * \param[in] dim        单个弱学习器分组中弱学习器的数量
 * \param[in] using_fold 表示是否将弱学习器系数并入弱学习器
 * \param[in] handles    弱学习器回调函数集合
 * \return 内存分配成功返回真，否则返回假
 */
bool mvec_ada_init(struct mvec_adaboost *ada, turn_t group_len,
		   mlabel_t dim, bool using_fold,
		   const struct wl_handles *handles);

/**
 * \brief 从文件中读取 mvec_adaboost
 * \details \copydetails mvec_ada_read_fn
 */
bool mvec_ada_read(struct mvec_adaboost *adaboost, FILE * file,
		   const struct wl_handles *handles);

/**
 * \brief 将 mvec_adaboost 结构体写入到文件
 * \details \copydetails mvec_ada_write_fn
 */
bool mvec_ada_write(const struct mvec_adaboost *adaboost, FILE * file,
		    const struct wl_handles *handles);

/**
 * \brief 对 mvec_adaboost 进行深度复制
 * \details \copydetails mvec_ada_copy_fn
 */
void *mvec_ada_copy(struct mvec_adaboost *dst,
		    const struct mvec_adaboost *src,
		    const struct wl_handles *handles);

/**
 * \brief mvec_adaboost 内存释放方法
 * \details \copydetails mvec_ada_free_fn
 */
void mvec_ada_free(struct mvec_adaboost *adaboost,
		   const struct wl_handles *handles);

#endif
