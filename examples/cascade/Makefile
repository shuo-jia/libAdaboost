# 编译器设置
CC = gcc

# 库文件目录
SrcPath = ../../src

# 头文件目录
Inc = -I .
Inc += $(addprefix -I , $(SrcPath))

# 源文件
Src = $(wildcard $(SrcPath)/*.c)
Src += $(wildcard $(SrcPath)/AlphaCalc/*.c)
Src += $(wildcard $(SrcPath)/WeakLearner/*.c)
Src += $(wildcard $(SrcPath)/WeakLearner/stump/*.c)
Src += $(wildcard $(SrcPath)/WeakLearner/constant/*.c)
Src += ./image.c

# 链接选项
Link = -lm
# 如果使用 image.h 的 jpeg 读取功能，则需添加 libjpeg 库，并取消注释该语句
# Link += -ljpeg

# 编译选项（开启调试）；鉴于级联分类器训练时间较长，可启用 -DLOG 选项在训练时输出信息
Opt = -g -DLOG
# 当使用 image.h 的 jpeg 读取功能时，需取消注释该语句 
# Opt += -DIMG_JPEG

all: cas_train cas_detect

cas_train: cas_train.c $(Src)
	$(CC) $(Inc) $^ $(Link) $(Opt) -o cas_train

cas_detect: cas_detect.c $(Src)
	$(CC) $(Inc) $^ $(Link) $(Opt) -o cas_detect
