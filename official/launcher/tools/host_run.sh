#!/bin/sh

DO_CMD=$@

RUN_DOKCER="sudo nvidia-docker run --rm --name lightning_run2 --shm-size=96g --ulimit memlock=-1 --ipc=host --cap-add=IPC_LOCK --net=host --ulimit stack=67108864 --privileged --entrypoint=/bin/sh $DOCKER_MOUNT_PATH -w $DOCKER_CODE_PATH $DOCKER_NAME -c bash"

docker run -it --name lee_cuda -v /home:/home -e TZ=Asia/Shanghai --ipc=host -u root -e  --ulimit core=-1 --security-opt seccomp=unconfined --network host -v /sys/kernel:/sys/kernel --privileged --privileged docker_name



if [ $# -lt 1 ]; then
    DO_CMD=$RUN_DOKCER
fi

workers=`sed -n '/^[^#]/p' launcher/host/nodes_list`

RunDocker()
{
    for WORKER in $workers
        do
            echo ""
            echo HOST $WORKER
            ssh $WORKER $DO_CMD &
        done
    wait
    return 0
}

doCommand()
{
    for WORKER in $workers
        do
            echo ""
            echo HOST $WORKER
            ssh $WORKER $DO_CMD
        done
    wait
    return 0
}

doCopy()
{
    for WORKER in $workers
        do
            echo ""
            echo CP $WORKER:${CLIENT_CODE_PATH}
            scp -r launcher $WORKER:${CLIENT_CODE_PATH}/
        done
    wait
    return 0
}

if [ $# -lt 1 ] && [ ! "$DO_CMD" ];
then
    echo "$0 cmd"
    exit
fi
echo "$0: $DO_CMD"

if [ $# -lt 1 ]; then
    RunDocker $DO_CMD
else
    if [ "$DO_CMD" == "copy" ]; then
        doCopy $DO_CMD
    else
        if [ "$DO_CMD" == "stop" ]; then
            DO_CMD="sudo docker stop lightning_run"
        fi
        if [ "$DO_CMD" == "list" ]; then
            DO_CMD="ls -l ${NSYS_OUT_PATH}"
        fi
        if [ "$DO_CMD" == "check" ]; then
            DO_CMD="ls -l ${NSYS_OUT_PATH}/${NSYS_OUT}"
        fi
        if [ "$DO_CMD" == "check_log" ]; then
            DO_CMD="cat ${NSYS_OUT_PATH}/${NSYS_OUT}/lightning_run_$(date +%Y-%m-%d).log"
        fi
        doCommand $DO_CMD
    fi
fi
echo "task done"
