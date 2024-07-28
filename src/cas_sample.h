#ifndef CAS_SAMPLE_H
#define CAS_SAMPLE_H
#include <stdbool.h>
#include "boost_cfg.h"
#include "cascade.h"
/**
 * \file cas_sample.h
 * \brief Cascade 级联分类器的样本集类型定义及函数声明
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-16
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 级联分类器的样本集类型
struct cas_sample {
	sample_t **X;		///< 积分图指针数组
	sample_t **X2;		///< （灰度值平方的）积分图指针数组
	label_t *Y;		///< 样本标签数组
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief 样本集初始化函数
 * \param[out] sp          待初始化的样本集地址
 * \param[in] img_size     样本尺寸，每个样本用一个正方形图像表示，此字段指定边长
 * \param[in] face         需生成的人脸样本数量
 * \param[in] non_face     需生成的非人脸样本数量
 * \param[in, out] args    用户自定义参数
 * \param[in] get_face     回调函数，用于获取人脸样本，args 将被传递给该函数
 * \param[in] get_non_face 回调函数，用于获取非人脸图片，args 将被传递给该函数
 * \return 成功则返回真，否则返回假
 */
bool init_samples(struct cas_sample *sp, imgsz_t img_size, num_t face,
		  num_t non_face, void *args, cas_face_fn get_face,
		  cas_non_face_fn get_non_face);

/**
* \brief 更新调整训练集和验证集
* \param[in, out] sp      已初始化的样本集地址（包括训练集和验证集）
* \param[in, out] m       指向样本集样本数量。函数运行后，样本集样本数量数量将会
* 			  减少，不被使用的样本将被释放
* \param[in, out] args    用户自定义参数
* \param[in] img_size     训练样本的尺寸
* \param[in] get_non_face 回调函数，用于获取非人脸图片，args 将被传递给该函数
* \param[in] cascade      指向当前训练的级联分类器
* \param[in] hl           指向 Adaboost 分类器回调函数集
*/
bool update_samples(struct cas_sample *sp, num_t * m, void *args,
		    imgsz_t img_size, cas_non_face_fn get_non_face,
		    const struct cascade *cascade,
		    const struct haar_ada_handles *hl);

/**
 * \brief 释放样本集内存
 * \param[in] sp -已初始化的样本集地址
 * \param[in] count -当前样本集内的样本总数（包括验证集、训练集）
 */
void free_samples(struct cas_sample *sp, num_t count);

/**
 * \brief 计算积分图
 * \param[in] m  图像高度
 * \param[in] n  图像宽度
 * \param[out] x 保存有 flt_t 型矩阵的灰度图像，积分图也将保存在此
 */
void intgraph(imgsz_t m, imgsz_t n, sample_t x[m][n]);

/**
 * \brief 计算积分图（对灰度值的平方累加）
 * \details \copydetails intgraph
 */
void intgraph2(imgsz_t m, imgsz_t n, sample_t x[m][n]);

#endif
