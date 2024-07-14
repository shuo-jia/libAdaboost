// 基于哈尔特征的 Adaboost 分类器--子类（增加训练方法）
// 主要实现 Paul Viola, Michael Jones 于 2001 年给出的方法
// （Robust Real-time Object Detection）
#ifndef HAAR_ADABOOST_H
#define HAAR_ADABOOST_H
#include "haar_base.h"

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/*
 * Adaboost 训练方法
 * adaboost: 指向未初始化的 struct haar_adaboost 结构体
 * d: 指向 AdaBoost 分类器的最小检测率
 * f: 指向 AdaBoost 分类器的最大假阳率
 * l: 验证集样本数量
 * m: 训练集样本数量
 * h: 图像高度
 * w: 图像宽度
 * X: 图像灰度值的积分图指针数组，每个指针指向 h*w 的图像区域
 * X2: 图像灰度值平方的积分图指针数组，每个指针指向 h*w 的图像区域
 * Y: 样本标签集，长度为 l + m
 * handles: 弱学习器回调函数集合
 * 返回值：成功则返回真，并设置 *d 为当前检测率，*f 为当前假阳率；否则返回假
 * 	注：运行过程未出错，但在验证集上假阳率仍小于 *f的情形，同样返回真。
 */
// 近似方法，利用不等式缩放获取系数 alpha
bool haar_ada_approx_train(struct haar_adaboost *adaboost, flt_t * d,
			   flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			   const sample_t * X[], const sample_t * X2[],
			   const label_t Y[], const struct wl_handles *handles);

// 数值方法，应用牛顿二分法求系数 alpha
bool haar_ada_newton_train(struct haar_adaboost *adaboost, flt_t * d,
			   flt_t * f, num_t l, num_t m, imgsz_t h, imgsz_t w,
			   const sample_t * X[], const sample_t * X2[],
			   const label_t Y[], const struct wl_handles *handles);

/*
 * 获取分类结果（用于图片）
 * adaboost: 已训练完毕的 AdaBoost 学习器；
 * h: 窗口高度
 * w: 窗口宽度
 * wid: 图像实际宽度
 * x: 积分图
 * x2: 灰度值平方的积分图
 * scale: 与训练图片相比的尺度放大倍数
 * handles: 弱学习器回调函数集合
 * 输出：输出分类结果（置信度）
 */
// （弱学习器系数不并入弱学习器）
flt_t haar_ada_h(const struct haar_adaboost *adaboost, imgsz_t h, imgsz_t w,
		 imgsz_t wid, const sample_t x[h][wid],
		 const sample_t x2[h][wid], flt_t scale,
		 const struct wl_handles *handles);

#endif
