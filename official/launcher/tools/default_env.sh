export SSH_USER_WORKER="root@172.21.17.9"
export CLIENT_CODE_PATH="/data0/code/lm_lighting" # 在这里配置使用的机器config
export DOCKER_CODE_PATH="/data/code/lm_lighting" # 在这里配置使用的机器config
export DOCKER_NAME='taco-train:cuda12.3--nemo-24.03-1.4_0622' # 在这里配置使用的镜像images信息
export DOCKER_VERSION="nemo24.03_0622" # 为使用的镜像提供一个简短的名称
export ENV_HW="launcher/env/H20.sh" # 在这里配置使用的机器config
export DOCKER_MOUNT_PATH="-v /data0:/data -v /dataset:/datasets -v /res_official:/res_official" # 在这里配置使用的机器config
export PATH_PR=""  # 在这里配置输出根路径
export RUNNER_PLACE="inner_run"  # host_run or inner_run

if [ "${RUNNER_TYPE}" == "mlperf" ]; then
    export TEST_ID_LIST='launcher/test_list/test_mlperf'
else
    if [ "${RUNNER_TYPE}" == "launcher" ]; then
        export TEST_ID_LIST='launcher/test_list/test_launcher'
    else
        export TEST_ID_LIST='need to add'
    fi
fi