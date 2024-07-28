#ifndef CASCADE_H
#define CASCADE_H
#include "adaboost.h"
#include "link_list.h"
/**
 * \file cascade.h
 * \brief 级联的(Cascade) adaboost 分类器类型定义及函数声明
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-15
 */

/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 表示一个矩形（正方形）
struct cas_rect {
	imgsz_t start_x;	///< 左上角横坐标
	imgsz_t start_y;	///< 左上角纵坐标
	imgsz_t len;		///< 边长（正方形）
};

/// 含有所检测目标的矩形框
struct cas_det_rect {
	struct cas_rect rect;	///< 矩形框的位置及大小
	flt_t confidence;	///< 置信度
};

/// 级联的 AdaBoost
struct cascade {
	struct link_list adaboost;	///< AdaBoost 链表
	imgsz_t img_size;	///< 训练所用图片大小（正方形边长）
	flt_t f_p_ratio;	///< 假阳率
	flt_t det_ratio;	///< 检测率
};

/**
 * \brief 回调函数类型：获取图片及人脸矩形框。
 * 	每次执行，都将获取一张包含人脸的图片及表示人脸位置的矩形框，训练时将从
 * 	该图片截取人脸，直至取得给定数量的人脸样本
 * \param[out] h        用于保存图片高度
 * \param[out] w        用于保存图片宽度
 * \param[out] rect     用于保存人脸框的位置及大小
 * \param[in, out] args 用户自定义的参数，用于保证可重入性
 * \return 成功则返回灰度图片地址，否则返回 NULL。灰度图片用一个一维数组表示，灰
 * 			度值矩阵按行优先存储
 */
typedef const unsigned char *(*cas_face_fn) (imgsz_t * h, imgsz_t * w,
					     struct cas_rect * rect,
					     void *args);

/**
 * \brief 回调函数类型，获取非人脸图片。
 *      每次执行，都将获取一张不包含人脸的图片，图片可循环使用。训练时将从该图
 *      片随机截取非人脸样本
 * \param[out] h        用于保存图片高度
 * \param[out] w        用于保存图片宽度
 * \param[out] id       用于保存当前图像 id，需保证图像 id 唯一
 * \param[in, out] args 用户自定义的参数，用于保证可重入性
 * \return 成功则返回灰度图片地址，否则返回 NULL。灰度图片用一个一维数组表示，灰度
 * 			值矩阵按行优先存储
 */
typedef const unsigned char *(*cas_non_face_fn) (imgsz_t * h, imgsz_t * w,
						 num_t * id, void *args);

/*******************************************************************************
 * 				    函数声明
 ******************************************************************************/
/**
 * \brief cascade 分类器训练
 * \param[out] cascade     要进行训练的级联学习器
 * \param[in] d            单个 Adaboost 分类器所允许的最小检测率
 * \param[in] f            单个 Adaboost 分类器所允许的最大假阳率
 * \param[in] F            级联分类器所允许的最大假阳率
 * \param[in] train_pct    训练集在样本中的占比，其余样本作为验证集
 * \param[in] face         人脸样本数量
 * \param[in] non_face     非人脸样本数量
 * \param[in] img_size     用作训练的图片尺寸
 * \param[in] args         用户自定义参数
 * \param[in] get_face     回调函数，获取人脸样本，参数 args 将被传递给该函数；
 * \param[in] get_non_face 回调函数，获取非人脸图片，参数 args 将被传递给该函数
 * \param[in] hl           级联分类器回调函数集
 * \return 训练成功返回真，否则返回假
 * 	注：训练过程未出错，但无法达到指定假阳率的情形，仍返回真
 */
bool cas_train(struct cascade *cascade, flt_t d, flt_t f, flt_t F,
	       flt_t train_pct, num_t face, num_t non_face, imgsz_t img_size,
	       void *args, cas_face_fn get_face, cas_non_face_fn get_non_face,
	       struct haar_ada_handles *hl);

/**
 * \brief 保存级联分类器的模型参数到文件
 * \param[in] cascade 要保存参数的级联分类器
 * \param[out] file   已打开文件的文件指针（可写，二进制形式）
 * \param[in] hl      Adaboost 相关回调函数集合
 * \return 如果成功写入，则返回真；否则返回假
 */
bool cas_write(const struct cascade *cascade, FILE * file,
	       const struct haar_ada_handles *hl);

/**
 * \brief 从文件中读取级联分类器的模型参数
 * \param[out] cascade 要读取参数的级联分类器（未初始化）
 * \param[in] file     已打开文件的文件指针（可读，二进制形式）
 * \param[in] hl       Adaboost 相关回调函数集合
 * \return 如果成功读取，则返回真；否则返回假
 */
bool cas_read(struct cascade *cascade, FILE * file,
	      const struct haar_ada_handles *hl);

/**
 * \brief 销毁级联分类器
 * \param[in] cascade 要释放内存的级联分类器
 * \param[in] hl      Adaboost 相关回调函数集合
 */
void cas_free(struct cascade *cascade, const struct haar_ada_handles *hl);

/**
 * \brief 计算两个矩形之间的重叠度
 * \param[in] rect1 已初始化的矩形结构体
 * \param[in] rect2 已初始化的矩形结构体
 * \return 返回重叠度（0~1）
 */
flt_t IoU(const struct cas_rect *rect1, const struct cas_rect *rect2);

/**
 * \brief 获取分类结果
 * \param[in] cascade 已训练完毕或已从文件中读取参数的级联分类器
 * \param[in] n       积分图边长
 * \param[in] wid     图像的实际宽度
 * \param[in] x       积分图
 * \param[in] x2      灰度值平方的积分图
 * \param[in] hl      Adaboost 相关回调函数集合
 * \return 输出分类结果（置信度）
 */
flt_t cas_h(const struct cascade *cascade, imgsz_t n, imgsz_t wid,
	    const flt_t x[n][wid], const flt_t x2[n][wid],
	    const struct haar_ada_handles *hl);

/**
 * \brief 多尺度、多位置扫描图像并返回下一目标所在的矩形框
 * \param[in] cascade   已训练的级联分类器
 * \param[in, out] rect 一个已初始化的矩形框，函数将从该位置、该尺度继续扫描
 * \param[in] delta     检测时窗口每次移动的像素数，将以1.25的倍数不断被放大
 * \param[in] h         图像高度
 * \param[in] w         图像宽度
 * \param[in] x         积分图
 * \param[in] x2        像素平方的积分图
 * \param[in] hl        Adaboost 相关回调函数集合
 * \return 如果检测到目标，则返回大于 0 的置信度，且 rect 被置为矩形框所在位置；
 * 	 否则返回 -1。rect、delta 可用于指示下次检测的初始位置以及移动距离
 */
flt_t cas_nextobj(const struct cascade *cascade, struct cas_rect *rect,
		  imgsz_t * delta, imgsz_t h, imgsz_t w, const flt_t x[][w],
		  const flt_t x2[][w], const struct haar_ada_handles *hl);

/**
 * \brief 多尺度、多位置扫描图像并返回检测到的所有目标（链表）
 * \param[in] cascade 已训练的级联分类器
 * \param[in] h       图像高度
 * \param[in] w       图像宽度
 * \param[in] img     已读入的灰度图片
 * \param[in] delta   检测时窗口每次移动的像素数
 * \param[in] hl      Adaboost 相关回调函数集合
 * \return 返回一个保存有目标所在矩形框的链表，链表的卫星数据为
 * 	 struct cas_det_rect 类型变量
 */
struct link_list cas_detect(const struct cascade *cascade, imgsz_t h,
			    imgsz_t w, unsigned char img[h][w], imgsz_t delta,
			    const struct haar_ada_handles *hl);

/**
 * 一个训练级联分类器（人脸检测器）的例子。
 * \note
 * 这里数据集采用 BioID 人脸数据集(https://www.bioid.com/face-database/)。\n
 * pgm 图像需解压至 BioID-FaceDatabase-V1.2 目录下。\n
 * 除此之外，还需将人脸框标注文件保存为 p_mark（见于示例）。
 *
 * \par 图像的处理模块（类型及函数声明）
 * image.h
 * \include image.h
 * \par 图像的处理模块（函数实现）
 * image.c
 * \include image.c
 * \note
 * 该模块提供了 jpeg 图像读取功能，但依赖于 libjpeg 库，默认情况下不编译 jpeg
 * 图片相关功能。\n
 * 如需使用，则需将 libjpeg 加入编译；并定义 IMG_JPEG 宏，如在 gcc中加入编译选项：
 * 	-DIMG_JPEG
 *
 * \par 配置文件修改
 * boost_cfg.h
 * \include cascade/boost_cfg.h
 *
 * \par 人脸检测
 * 读取二进制文件 cascade_data（保存有级联分类器），简单地将人脸框输出到控制台
 * \include cas_detect.c
 *
 * \par 人脸检测器训练
 * 使用人脸灰度图像来训练级联分类器（pgm 图片格式），并将级联分类器保存到二进制文
 * 件 cascade_data 中。\n
 * 训练时，每张图片以人脸为中心，分为上、下两张子图，以及左、右子图，共计 4 张
 * 子图，作为非人脸图像。
 * \example cas_train.c
 *
 * \example p_mark
 * \par 附录：BioID 数据集人脸位置及大小的标注文件。
 * 文件每一行的第一个字段给出 BioID 数据集图像的文件名，第二个字段给出人脸左上
 * 角的横坐标，第三个字段给出人脸左上角的纵坐标，第四个字段给出人脸框的高度，第
 * 五个字段给出人脸框的宽度。这里高度与宽度是相等的。
 */
#endif
