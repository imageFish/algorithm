import requests
import json
import boto3

response = requests.get('https://example.com/server/uploadService.php')
# 请替换为您服务端部署的 临时密钥和上传信息获取服务 的地址，这里的参数是举例，请根据实际情况，使用您自己的业务参数，
# 然后服务端根据当前用户的登录情况、业务实际情况进行临时密钥的授权（比如是否允许上传、允许上传到哪个文件等）

ret = response.json()

s3 = boto3.client(
    's3',
    aws_access_key_id=ret['credentials']['accessKeyId'],
    aws_secret_access_key=ret['credentials']['secretAccessKey'],
    aws_session_token=ret['credentials']['sessionToken'],
    endpoint_url=ret['s3Endpoint']
)

