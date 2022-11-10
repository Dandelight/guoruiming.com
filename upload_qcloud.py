# -*- coding=utf-8
from qcloud_cos import (CosConfig,
    CosS3Client,
    CosServiceError,
    CosClientError
)
from qcloud_cos.cos_threadpool import SimpleThreadPool

import fnmatch
import logging
from typing import (
    List,
)

import sys
import os

# 敏感信息分离
from dotenv import load_dotenv

load_dotenv()

# 正常情况日志级别使用INFO，需要定位时可以修改为DEBUG，此时SDK会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, filename='upload.log', filemode='a')

# 设置用户属性, 包括 secret_id, secret_key, region等。Appid 已在CosConfig中移除，请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成
secret_id = os.getenv("QCLOUD_COS_SECRET_ID")     # 替换为用户的 SecretId，请登录访问管理控制台进行查看和管理，https://console.cloud.tencent.com/cam/capi
secret_key = os.getenv("QCLOUD_COS_SECRET_KEY")   # 替换为用户的 SecretKey，请登录访问管理控制台进行查看和管理，https://console.cloud.tencent.com/cam/capi
region = os.getenv("QCLOUD_COS_REGION")      # 替换为用户的 region，已创建桶归属的region可以在控制台查看，https://console.cloud.tencent.com/cos5/bucket
                           # COS支持的所有region列表参见https://cloud.tencent.com/document/product/436/6224
token = None               # 如果使用永久密钥不需要填入token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见https://cloud.tencent.com/document/product/436/14048
bucket = os.getenv("QCLOUD_COS_BUCKET")
uploadDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
cosBase = "/"
incremental = bool(os.getenv("BLOG_BUILD_INCREMENTAL"))

config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token)  # 获取配置对象

def get_files(dir):
    """Walk the `docs_dir` and return a Files collection."""
    files = []
    exclude = ['.*', '/templates']

    for source_dir, dirnames, filenames in os.walk(dir, followlinks=True):
        relative_dir = os.path.relpath(source_dir, dir)

        for dirname in list(dirnames):
            path = os.path.normpath(os.path.join(relative_dir, dirname))
            # Skip any excluded directories
            if _filter_paths(basename=dirname, path=path, is_dir=True, exclude=exclude):
                dirnames.remove(dirname)
        dirnames.sort()

        for filename in _sort_files(filenames):
            path = os.path.normpath(os.path.join(relative_dir, filename))
            # Skip any excluded files
            if _filter_paths(basename=filename, path=path, is_dir=False, exclude=exclude):
                continue
            files.append(path)
    return files


def _sort_files(filenames) -> List[str]:
    """Always sort `index` or `README` as first filename in list."""

    def key(f):
        if os.path.splitext(f)[0] in ['index', 'README']:
            return (0,)
        return (1, f)

    return sorted(filenames, key=key)


def _filter_paths(basename: str, path: str, is_dir: bool, exclude) -> bool:
    """.gitignore style file filtering."""
    for item in exclude:
        # Items ending in '/' apply only to directories.
        if item.endswith('/') and not is_dir:
            continue
        # Items starting with '/' apply to the whole path.
        # In any other cases just the basename is used.
        match = path if item.startswith('/') else basename
        if fnmatch.fnmatch(match, item.strip('/')):
            return True
    return False

def do_upload(config: CosConfig):
    client = CosS3Client(config)
    g = os.walk(uploadDir)
    # 创建上传的线程池
    pool = SimpleThreadPool()
    for file in get_files(uploadDir):
        srcKey = os.path.join(uploadDir, file)
        cosObjectKey = (cosBase + file.replace('\\', '/')).strip('/')
        pool.add_task(client.upload_file, bucket, cosObjectKey, srcKey)

    pool.wait_completion()
    result = pool.get_result()
    if not result['success_all']:
        logging.warning("Not all files upload sucessed. you should retry")
    else:
        logging.info("All files uploaded successfully.")

if __name__ == '__main__':
    try:
        do_upload(config)
    except CosClientError:
        logging.error("Client Error: {}".format(CosClientError))
    except CosServiceError:
        logging.error("Server Error: {}".format(CosServiceError))
