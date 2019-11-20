docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/textgen-docker:$1 -f docker/Dockerfile .

docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/textgen-docker:$1