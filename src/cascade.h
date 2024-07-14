/* cascade.h --- 级联的(Cascade) adaboost 分类器 */
#ifndef CASCADE_H
#define CASCADE_H
#include "adaboost.h"
#include "link_list.h"

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
// 表示一个矩形
struct cas_rect {
	imgsz_t start_x;		// 左上角横坐标
	imgsz_t start_y;		// 左上角纵坐标
	imgsz_t height;			// 矩形高度
	imgsz_t width;			// 矩形宽度
};

// 含有所检测目标的矩形框
struct cas_det_rect {
	struct cas_rect rect;		// 矩形框的位置及大小
	flt_t confidence;		// 置信度
};

// 级联的 AdaBoost
struct cascade {
	struct link_list adaboost;	// AdaBoost 链表
	imgsz_t img_size;		// 训练所用图片大小（正方形边长）
	flt_t f_p_ratio;		// 假阳率
	flt_t det_ratio;		// 检测率
};

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/* 
 * 进行训练
 * cascade: 要进行训练的级联学习器
 * d: 单个 Adaboost 分类器所允许的最小检测率
 * f: 单个 Adaboost 分类器所允许的最大假阳率
 * F: 级联分类器所允许的最大假阳率
 * l: 验证集样本数量
 * m: 训练集样本数量
 * n: 正方形窗口的边长，即训练图像的长和宽
 * X: 积分图数组，前 l 个为验证集，后 m 个为训练集
 * X2: 灰度值平方的积分图数组，前 l 个为验证集，后 m 个为训练集
 * Y: 样本对应的分类
 * hl: Adaboost 相关回调函数集合
 * 返回值：如果学习算法运行成功，则返回真；运行失败则返回假。
 */
bool cas_train(struct cascade *cascade, flt_t d, flt_t f, flt_t F, num_t l,
	       num_t m, imgsz_t n, sample_t * const X[], sample_t * const X2[],
	       const label_t Y[], const struct haar_ada_handles *hl);

/*
 * 将级联分类器 src 追加到级联分类器 dst 之后，训练图片大小必须相同
 * dst: 目标级联分类器，追加成功后 dst 的检测率、假阳率字段不保证精确
 * src: 源级联分类器，追加成功后该级联分类器被初始化为空
 * 返回值：成功则返回真，否则返回假
 */
bool cas_cat(struct cascade *dst, struct cascade *src);

/* 
 * 保存级联分类器的模型参数到文件
 * cascade: 要保存参数的级联分类器
 * file: 已打开文件的文件指针（可写，二进制形式）
 * hl: Adaboost 相关回调函数集合
 * 返回值：如果成功写入，则返回真；否则返回假
 */
bool cas_write(const struct cascade *cascade, FILE * file,
	       const struct haar_ada_handles *hl);

/* 
 * 从文件中读取级联分类器的模型参数
 * cascade: 要读取参数的级联分类器（已初始化），除必要的回调函数外均被覆盖
 * file: 已打开文件的文件指针（可读，二进制形式）
 * hl: Adaboost 相关回调函数集合
 * 返回值：如果成功读取，则返回真；否则返回假
 */
bool cas_read(struct cascade *cascade, FILE * file,
	      const struct haar_ada_handles *hl);

/* 
 * 销毁级联分类器
 * cascade: 要释放内存的级联分类器
 * hl: Adaboost 相关回调函数集合
 */
void cas_free(struct cascade *cascade, const struct haar_ada_handles *hl);

/*
 * 计算积分图
 * m: 图像高度
 * n: 图像宽度
 * x: 保存有 flt_t 型矩阵的灰度图像，积分图也将保存在此
 */
void cas_intgraph(imgsz_t m, imgsz_t n, flt_t x[m][n]);

/*
 * 计算积分图（对灰度值的平方累加）
 * m: 图像高度
 * n: 图像宽度
 * x: 保存有 flt_t 型矩阵的灰度图像，积分图也将保存在此
 */
void cas_intgraph2(imgsz_t m, imgsz_t n, flt_t x[m][n]);

/*
 * 计算两个矩形之间的重叠度
 * rect1: 已初始化的矩形结构体
 * rect2: 已初始化的矩形结构体
 * 返回值：返回重叠度（0~1）
 */
flt_t IoU(const struct cas_rect *rect1, const struct cas_rect *rect2);

/* 
 * 获取分类结果
 * cascade: 已训练完毕或已从文件中读取参数的级联分类器
 * n: 积分图边长
 * wid: 图像的实际宽度
 * x: 积分图
 * x2: 灰度值平方的积分图
 * hl: Adaboost 相关回调函数集合
 * 返回值：输出分类结果（置信度）
 */
flt_t cas_h(const struct cascade *cascade, imgsz_t n, imgsz_t wid,
	    const flt_t x[n][wid], const flt_t x2[n][wid],
	    const struct haar_ada_handles *hl);

/*
 * 多尺度、多位置扫描图像并返回下一目标所在的矩形框
 * cascade: 已训练的级联分类器
 * rect: 一个已初始化的矩形框，函数将从该位置、该尺度继续扫描
 * delta: 检测时窗口每次移动的像素数，将以1.25的倍数不断被放大
 * h: 图像高度
 * w: 图像宽度
 * x: 积分图
 * x2: 像素平方的积分图
 * hl: Adaboost 相关回调函数集合
 * 返回值：如果检测到目标，则返回大于 0 的置信度，且 rect 被置为矩形框所在位置；
 * 	 否则返回 -1。rect、delta 可用于指示下次检测的初始位置以及移动距离
 */
flt_t cas_nextobj(const struct cascade *cascade, struct cas_rect *rect,
		  imgsz_t * delta, imgsz_t h, imgsz_t w, const flt_t x[][w],
		  const flt_t x2[][w], const struct haar_ada_handles *hl);

/*
 * 多尺度、多位置扫描图像并返回检测到的所有目标（链表）
 * cascade: 已训练的级联分类器
 * h: 图像高度
 * w: 图像宽度
 * img: 已读入的灰度图片
 * delta: 检测时窗口每次移动的像素数
 * hl: Adaboost 相关回调函数集合
 * 返回值：返回一个保存有目标所在矩形框的链表，链表的卫星数据为
 * 	 struct cas_det_rect 类型变量
 */
struct link_list cas_detect(const struct cascade *cascade, imgsz_t h,
			    imgsz_t w, unsigned char img[h][w], imgsz_t delta,
			    const struct haar_ada_handles *hl);

#endif
