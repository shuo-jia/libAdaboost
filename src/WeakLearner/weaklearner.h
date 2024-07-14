#ifndef WEAKLEARNER_H
#define WEAKLEARNER_H
#include <stdio.h>
#include <stdbool.h>
#include "boost_cfg.h"
/**
 * \file weaklearner.h
 * \brief 弱学习器函数调用集定义及函数声明。
 * 	此文件定义了与弱学习器相关的一组回调函数集合，使用类似工厂模式的方法设
 * 	置不同种类的回调函数集。此文件为不同弱学习器提供了一组通用的接口。
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-14
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/**
 * \brief 回调函数类型：输出分类结果（输入为样本向量，输出结果不带置信度）
 * \param[in] stump: 已保存训练结果的决策树桩
 * \param[in] x: 样本向量
 * \param[in] n: 样本向量的长度
 * \return 返回输出值（-1、+1）
 */
typedef label_t(*wl_h_vec_fn) (const void *stump, const sample_t x[], dim_t n);

/**
 * \brief 回调函数类型：输出分类结果（输入为样本向量，输出结果带置信度）
 * \param[in] stump: 已保存训练结果的决策树桩
 * \param[in] x: 样本向量
 * \param[in] n: 样本向量的长度
 * \return 返回输出值（置信度）
 */
typedef flt_t(*wl_h_vec_cf_fn) (const void *stump, const sample_t x[], dim_t n);

/**
 * \brief 回调函数类型：输出分类结果（输入为图像某一矩形区域左上角的地址，输出
 * 	结果不带置信度）
 * \param[in] stump 已训练完毕的决策树桩
 * \param[in] h     窗口高度
 * \param[in] w     窗口宽度
 * \param[in] wid   图像实际宽度
 * \param[in] x     积分图（h * w 大小的二维数组）
 * \param[in] x2    灰度值平方的积分图（h * w 大小的二维数组）
 * \param[in] scale 与训练图片相比的尺度放大倍数
 * \return 输出分类结果
 */
typedef label_t(*wl_h_haar_fn) (const void *stump, imgsz_t h, imgsz_t w,
				imgsz_t wid, const sample_t x[h][wid],
				const sample_t x2[h][wid], flt_t scale);

/**
 * \brief 回调函数类型：输出分类结果（输入为图像某一矩形区域左上角的地址，输出
 * 	结果带置信度）
 * \details \copydetails haar_stump_cf_h()
 */
typedef flt_t(*wl_h_haar_cf_fn) (const void *stump, imgsz_t h, imgsz_t w,
				 imgsz_t wid, const sample_t x[h][wid],
				 const sample_t x2[h][wid], flt_t scale);

/**
 * \brief 回调函数类型：对样本进行训练（输入为样本向量构成的矩阵，成功则返回真）
 * \param[out] stump 未初始化的决策树桩
 * \param[in] m      样本数量
 * \param[in] n      样本特征数量
 * \param[in] X      样本集
 * \param[in] Y      样本对应标签（1 或 -1 构成的数组）
 * \param[in] D      样本概率分布
 * \param[in] cache  缓存指针，可使用 vec_new_cache() 创建
 * \return 成功则返回真；失败则返回假
 */
typedef bool (*wl_train_vec_fn)(void *stump, num_t m, dim_t n,
				const sample_t X[m][n], const label_t Y[],
				const flt_t D[], const void *cache);

/**
 * \brief 回调函数类型：对样本进行训练（输入为样本的指针数组，每个样本用长度
 * 	为 h*w 的数组表示）
 * \param[out] stump 未初始化的决策树桩
 * \param[in] m     样本数量
 * \param[in] h     窗口高度
 * \param[in] w     窗口宽度
 * \param[in] X     积分图数组（每个元素指向 h * w 大小的二维数组）
 * \param[in] X2    灰度值平方的积分图数组（每个元素指向 h * w 大小的二维数组）
 * \param[in] Y     样本标签
 * \param[in] D     样本概率分布数组
 * \return 成功则返回真，否则返回假
 */
typedef bool (*wl_train_haar_fn)(void *stump, num_t m, imgsz_t h, imgsz_t w,
				 const sample_t * X[], const sample_t * X2[],
				 const label_t Y[], const flt_t D[]);

/**
 * \brief 回调函数类型：从文件中读取弱学习器
 * \param[out] stump 未初始化的决策树桩
 * \param[in] file   已打开的文件
 * \return 成功则返回真；失败则返回假
 */
typedef bool (*wl_read_fn)(void *stump, FILE * file);

/**
 * \brief 回调函数类型：将弱学习器写入到文件
 * \param[in] stump 已初始化的决策树桩（保存训练结果）
 * \param[out] file 已打开的文件
 * \return 成功则返回真；失败则返回假
 */
typedef bool (*wl_write_fn)(const void *stump, FILE * file);

/**
 * \brief 回调函数类型：对弱学习器进行深度复制
 * \param[out] dst 未初始化的决策树桩
 * \param[in] src  已初始化的决策树桩，内容将被复制到 dst
 * \return 成功则返回 dst；失败则返回 NULL
 */
typedef void *(*wl_copy_fn)(void *dst, const void *src);

/**
 * \brief 回调函数类型：释放弱学习器内存空间
 * \param[in] stump 已申请内存的决策树桩（训练或从文件读取决策树桩）
 */
typedef void (*wl_free_fn)(void *stump);

/// 弱学习器回调函数集合
struct wl_handles {
	size_t size;		///< 弱学习器类型的大小（字节），
				/**< 在不使用缓存时可忽略该字段 */
	bool using_confident;	///< 表示是否使用置信度，使用置信
				/**< 度则置为真，否则为假 */
	union {
		wl_h_vec_fn vec;
		wl_h_vec_cf_fn vec_cf;
		wl_h_haar_fn haar;
		wl_h_haar_cf_fn haar_cf;
	} hypothesis;		///< 输出弱学习器分类结果
	union {
		wl_train_vec_fn vec;
		wl_train_haar_fn haar;
	} train;		///< 弱学习器训练
	wl_read_fn read;	///< 从文件中读取弱学习器
	wl_write_fn write;	///< 将弱学习器写入到文件
	wl_copy_fn copy;	///< 对弱学习器进行深度复制
	wl_free_fn free;	///< 释放弱学习器内存空间
};

/*******************************************************************************
 * 				    函数原型
 ******************************************************************************/
/**
 * \brief 将回调函数集设为常量决策树桩，仅用于测试。
 * \param[out] handles 回调函数结构体地址。部分回调函数不存在，则对应成员设为
 * 	NULL
 */
void wl_set_constant(struct wl_handles *handles);

/**
 * \brief 将回调函数集设为处理向量的决策树桩，不带置信度，变量为连续变量
 * \details \copydetails wl_set_constant()
 */
void wl_set_vec_cstump(struct wl_handles *handles);

/**
 * \brief 将回调函数集设为处理向量的决策树桩，带置信度，变量为连续变量
 * \details \copydetails wl_set_constant()
 */
void wl_set_vec_cstump_cf(struct wl_handles *handles);

/**
 * \brief 将回调函数集设为处理向量的决策树桩，不带置信度，变量为离散变量
 * \details \copydetails wl_set_constant()
 */
void wl_set_vec_dstump(struct wl_handles *handles);

/**
 * \brief 将回调函数集设为处理向量的决策树桩，带置信度，变量为离散变量
 * \details \copydetails wl_set_constant()
 */
void wl_set_vec_dstump_cf(struct wl_handles *handles);

/**
 * \brief 将回调函数集设为 Haar 决策树桩，不带置信度
 * \details \copydetails wl_set_constant()
 */
void wl_set_haar(struct wl_handles *handles);

/**
 * \brief 将回调函数集设为 Haar 决策树桩，带置信度
 * \details \copydetails wl_set_constant()
 */
void wl_set_haar_cf(struct wl_handles *handles);

/**
 * \brief 将回调函数集设为 Haar 决策树桩，使用进化算法进行训练，不带置信度
 * \details \copydetails wl_set_constant()
 */
void wl_set_haar_ga(struct wl_handles *handles);

/**
 * \brief 将回调函数集设为 Haar 决策树桩，使用进化算法进行训练，带置信度
 * \details \copydetails wl_set_constant()
 */
void wl_set_haar_ga_cf(struct wl_handles *handles);

#endif
