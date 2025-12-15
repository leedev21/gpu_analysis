# cudahook

## Requirements
```sh
pip install -r requirements.txt
```


## Build
```sh
cd cudahook
./build.sh
```
目前输出路径
```
cudahook/cmake_build/bin
```

## Run
1. 设置环境变量 LD_LIBRARY_PATH
2. 执行程序

```sh
export LD_LIBRARY_PATH=path_to_cudahook/cmake_build/bin:$LD_LIBRARY_PATH
youre_test.sh
```
 

## Note
1. 目前只支持在安装 cuda 12 环境。
2. cudahook 支持2种工作模式，native 模式和 dry-run 模式，其中 native模式为不影响原有执行过程，只会按需要输出cuda api 相关执行日志； dry-run模式下会屏蔽相关的cuda api 执行过程。
3. 环境变量 CH_WORK_MODE_NATIVE_ENABLE 设置为1 对应native模式，设置为0，对应dry-run模式。 默认CH_WORK_MODE_NATIVE_ENABLE设置为 1。