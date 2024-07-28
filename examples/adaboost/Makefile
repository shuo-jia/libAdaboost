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
Src += ./table.c

# 链接选项
Link = -lm

# 编译选项（开启调试）
Opt = -g

all: ada_train mada_train

ada_train: ada_train.c $(Src)
	$(CC) $(Inc) $^ $(Link) $(Opt) -o ada_train
mada_train: mada_train.c $(Src)
	$(CC) $(Inc) $^ $(Link) $(Opt) -o mada_train
clean:
	rm ada_train mada_train
