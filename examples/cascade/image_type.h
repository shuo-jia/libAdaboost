#ifndef IMAGE_TYPE_H
#define IMAGE_TYPE_H

/**
 * \file image_type.h
 * \brief 图像类型定义
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-27
 */
/*******************************************************************************
 * 				    类型定义
 ******************************************************************************/
/// 图像结构体，表示一张图片
struct image {
	int height;			///< 图像高度
	int width;			///< 图像宽度
	int depth;			///< 图像深度
	unsigned char img[];		/** 图像数组，每 depth 个字节表示一个
					  像素 */
};

/// 表示一个矩形
struct rectangle {
	int start_x;			///< 矩形左上角横坐标
	int start_y;			///< 矩形左上角纵坐标
	int height;			///< 矩形高度
	int width;			///< 矩形宽度
};

/*******************************************************************************
 * 				   宏函数定义
 ******************************************************************************/
/// 类型转换：将图像（struct image *）转换为三维数组（高 * 宽 * 深度）
#define IMAGE_PTR(pic) ((unsigned char (*)[(pic)->width][(pic)->depth]) (pic)->img)
/// 声明语句：将 ptr 变量声明为适合 IMAGE_PTR(pic) 的三维数组指针（高 * 宽 * 深度）
#define IMAGE_PTR_TYPE(ptr, pic) unsigned char (*ptr)[(pic)->width][(pic)->depth]

#endif
