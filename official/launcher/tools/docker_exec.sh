CMD="\"cd $DOCKER_CODE_PATH && pwd && source ./$RUN_CONFIG && source ./$ENV_HW && $EXP_CONFIG bash ./$RUN_CMD\""
export RUN_DOKCER="sudo nvidia-docker run --rm --name lightning_run2 --shm-size=96g --ulimit memlock=-1 --ipc=host --cap-add=IPC_LOCK --net=host --ulimit stack=67108864 --privileged --entrypoint=/bin/sh $DOCKER_MOUNT_PATH -w $DOCKER_CODE_PATH $DOCKER_NAME -c bash"
