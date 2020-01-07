NAMESPACE=("shuzhi-amd64")
for i in ${NAMESPACE[*]}
do
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/textgen-docker:$1 -f docker/docker/Dockerfile .

    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/textgen-docker:$1
done