# from datasets import load_dataset
# import os
#
# # 设置数据集下载路径
# dataset_path = "/home/syt/project/compressor_500/data"  # 替换为您想要的路径
#
# # 设置环境变量使用国内镜像
# # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  export HF_ENDPOINT=https://hf-mirror.com
#
# print(dataset_path)
# # 下载数据集到指定位置
# ds = load_dataset(
#     "DKYoon/SlimPajama-6B",
#     cache_dir=dataset_path,
#     trust_remote_code=True
# )
#
# print("数据集已下载到:", dataset_path)
# print("数据集信息:", ds)


"""
from datasets import load_dataset

ds = load_dataset("mrqa-workshop/mrqa")

"""


from datasets import load_dataset
import os

# 设置数据集下载路径
dataset_path = "/home/syt/project/compressor_500/data"  # 替换为您想要的路径

# 设置环境变量使用国内镜像
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  export HF_ENDPOINT=https://hf-mirror.com

print(dataset_path)
# 下载数据集到指定位置
ds = load_dataset(
    "mrqa-workshop/mrqa",
    cache_dir=dataset_path,
    trust_remote_code=True
)

print("数据集已下载到:", dataset_path)
print("数据集信息:", ds)


