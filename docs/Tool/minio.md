Minio 是一款全开源，Amazon S3 兼容的对象存储服务器软件。

```shell
docker run \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  --rm -d \
  -v $HOME/minio/data:/data \
  -v $HOME/minio/config:/root/.minio \
  -e "MINIO_ROOT_USER=IHCVodsfa23jsksssv12" \
  -e "MINIO_ROOT_PASSWORD=IUH2ioh23xxIid33" \
  -e MINIO_SERVER_URL=http://localhost:9000 \
  quay.io/minio/minio server /data \
  --address ":9000" --console-address ":9001"
```
