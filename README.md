\mainpage 使用说明
# libAdaboost
libAdaboost 使用 C 语言实现了 AdaBoost 分类器和级联分类器，能被应用于二分类、多
分类任务。算法完全由 C 语言实现，便于移植到其他平台。

# 使用方法
使用前，将 src/ 目录及其子目录加入工程；至多只须引入 adaboost.h 及 cascade.h 两
个头文件。

## AdaBoost 二分类任务
在源代码中包含头文件 adaboost.h，使用 ada_set_vec() 函数设置 AdaBoost 采用的方法。
如
```c
#include "adaboost.h"
int main (void)
{
	struct vec_adaboost adaboost;
	struct vec_ada_handles handles;

	ada_set_vec(&handles, ADA_FOLD, ADA_NO_CONFIDENT, ADA_DISCRETE);
	if (!handles.train(&adaboost, 100, m, n, X, Y, true, &handles.wl_hl)) {
		fprintf(stderr, "Training Error.\n");
		exit(EXIT_FAILURE);
	}

	...

	handles.free(&adaboost, &handles.wl_hl);
	return 0;
}
```
这将对样本集 {X, Y} 进行训练，训练方法由 handles 变量决定。具体细节见示例程序
及 adaboost.h 头文件。

## AdaBoost 多分类任务
在源代码中包含头文件 adaboost.h，使用 ada_set_mvec() 函数设置 AdaBoost 采用的方法。
如
```c
#include "adaboost.h"
int main (void)
{
	struct mvec_adaboost adaboost;
	struct mvec_ada_handles handles;

	ada_set_mvec(&handles, ADA_HLOSS, ADA_FOLD, ADA_CONTINUOUS);
	if (!handles.train(&adaboost, 100, m, n, X, Y, true, &handles.wl_hl)) {
		fprintf(stderr, "Training Error.\n");
		exit(EXIT_FAILURE);
	}

	...
	
	handles.free(&adaboost, &handles.wl_hl);
	return 0;
}
```
具体细节见示例程序及 adaboost.h 头文件。

## 级联分类器

在源代码中包含头文件 cascade.h，使用 ada_set_haar() 函数设置 AdaBoost 采用的方法。
如
```c
#include "cascade.h"
int main (void)
	struct cascade cascade;
	struct haar_ada_handles handles;
	ada_set_haar (&handles, ADA_ASYM_IMP, ADA_OPT);

	// 下方 cas_train() 略去了部分参数
	if (!cas_train (&cascade, ..., &handles)) {
		fprintf(stderr, "Training Error.\n");
		exit(EXIT_FAILURE);
	}

	...

	cas_free (&cascade, &handles);
	return 0;
}
```
具体细节见示例程序及 adaboost.h，cascade.h 头文件。

# 示例程序
example/ 目录给出了一些示例程序。

其中，example/adaboost/ 使用手写字符数据集进行训练，给出了一个二分类任务的示例
程序以及一个多分类任务的示例程序；

example/cascade/ 使用 BioID 人脸数据集进行训练，给出了一个用于人脸检测的级联分
类器示例程序。

# 文档生成
libAdaboost 采用了 Doxygen 注释规范，可通过 Doxygen 工具生成代码文档。安装
Doxygen 后，在 doc/ 目录下执行命令：
```
doxygen
```
将生成 Html 文档及 LaTex 文档，LaTex 文档可使用 pdflatex 编译输出为 pdf 文件。
