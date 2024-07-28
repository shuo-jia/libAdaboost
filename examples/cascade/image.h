#ifndef IMAGE_H
#define IMAGE_H
#include <stdbool.h>
#include "image_type.h"

/**
 * \file image.h
 * \brief 图像类型函数声明。
 * \author Shuojia
 * \version 1.0
 * \date 2024-07-27
 */
/*******************************************************************************
 * 				    函数原型
 ******************************************************************************/
/**
 * \brief 返回一个指定大小的图像结构体
 * \param[in] height 图像高度
 * \param[in] width  图像宽度
 * \param[in] depth  图像深度
 * \return 操作成功则返回指定图像大小的图像结构体指针；失败则返回 NULL
 */
struct image * new_image (int height, int width, int depth);

#ifdef IMG_JPEG
/**
 * \brief 读取 jpeg 图片。在定义了 IMG_JPEG 宏的情况下才开启 jpeg 图片的支持
 * \param[in] name jpeg 图片的路径
 * \return 操作成功则返回一个图像结构体指针，且该结构体保存有解码后的 jpeg 图像；
 * 	失败则返回 NULL
 */
struct image * imread_jpeg (const char * name);
#endif

/**
 * \brief 读取 pgm（灰度）图片
 * \param[in] name pgm 图片的路径
 * \return 操作成功则返回一个图像结构体指针，且该结构体保存有解码后的 pgm 图像；
 * 	失败则返回 NULL
 */
struct image * imread_pgm (const char * name);

/**
 * \brief 返回一张子图
 * \param[in] img  指向已初始化的图像结构体
 * \param[in] rect 指向已初始化的矩形结构体，指示子图的位置和大小
 * \return 返回一个图像结构体指针，指向子图；失败则返回 NULL
 */
struct image * get_sub_image (const struct image *img, const struct rectangle *rect);

/**
 * \brief 图像裁剪操作
 * \param[in, out] img 指向已初始化的图像结构体，将在该图片上执行裁剪操作
 * \param[in] rect     指向已初始化的矩形结构体，在该矩形区域内的图像将被保留
 */
void cropping_image (struct image *img, const struct rectangle *rect);

/**
 * \brief 图像缩小操作
 * \param[in, out] img 指向已初始化的图像结构体，将在该图片上进行缩小操作
 * \param[in] height   缩小后的图片高度
 * \param[in] width    缩小后的图片宽度
 */
void reduced_image (struct image *img, int height, int width);

/**
 * \brief 灰度图转换操作，将图片转换为灰度图片（即深度为 1）
 * \param[in, out] img 待转换图片，转换后的图片将保存在此处
 */
void make_grey_image (struct image *img);

/**
 * \brief 调整存储空间使得刚好能容纳图片。在对图片进行运算时，凡导致图像大小减
 * 	小的运算都不会减小实际内存空间，可对最终的图片执行此函数以减小内存空间
 * \param[in, out] img 指向已初始化的图像结构体
 * \return 如果操作成功，则返回新的内存地址；否则，返回 img
 */
struct image * fit_image (struct image *img);

/**
 * \brief 均值滤波操作
 * \param img  指向已初始化的图像结构体
 * \param size 表示滤波模板的尺寸
 * \return 返回一个新的图像结构体地址，该图像结构体保存有已执行滤波操作的图像；
 * 	如果操作失败，则返回 NULL
 */
struct image * mean_filter (const struct image *img, int size);

#endif
