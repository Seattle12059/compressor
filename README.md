# 500xCompressor-2nd

**这里是500xCompressor 第二次前向不使用UPL的消融实验**

即,第一次前向**传入**`position_id`，第二次前向**不传入**`position_id`

注意:**所有操作和主分支是一样的**。

与主分支的区别是我们修改了代码：请看commit：`when use_pe is true, don't pass position_id in 2nd forward pass `

运行UPL的实验即可。

配置文件在`./experiment/500x_Llama-3.2-1B_UPL_ablation-2nd`

**配置文件夹建议放在local_experiment文件下**,否则无法切换分支。

```
mkdir ./experiment/local_experiment
cp -r ./experiment/500x_Llama-3.2-1B_UPL_ablation-2nd ./experiment/local_experiment/500x_Llama-3.2-1B_UPL_ablation-2nd
```


----

其他实验在其他分支里面，你可以用下面命令查看并切换到其他分支
```
git branch -a
git checkout <分支名>
```

操作与主分支一致，但**需要回退transformers版本来使用旧版的KVcache.**
`
pip install transformers==4.46.3 --force-reinstall
`


1. 项目文件夹简称UPL，首先安装python及其需要的包运行以下命令
```
git clone 仓库名
cd 仓库名
conda create -n UPL python==3.10.4
conda activate UPL
pip install -r requirements.txt
```
2. 安装好环境后配置所需模型、数据集
-  模型：https://huggingface.co/meta-llama/Llama-2-7b-hf
-  预训练数据集：https://huggingface.co/datasets/DKYoon/SlimPajama-6B
-  微调数据集：https://huggingface.co/datasets/mrqa-workshop/mrqa

3. 修改experiment文件夹下的[config.json](./experiment/main/500x_Llama-3.2-1B_DPL/config.json)文件，填写模型和数据集的本地路径
```
"model_id": "your_model_path",
"dataset_repo": "your_data_path/DKYoon/SlimPajama-6B",
"instruction_dataset_repo": "your_data_path/mrqa-workshop_mrqa"
```

4. 修改experiment文件夹下的[config.json](./experiment/main/500x_Llama-3.2-1B_DPL/config.json)文件，**根据您的GPU数量修改梯度累积的步数**，确保 
```
batch_size_per_device*device_count*gradient_accumulation_steps==total_batch_size
```
```
"batch_size_per_device": 1,
"device_count": 8,
"gradient_accumulation_steps": 2,
```

## 继续预训练
```
cd pretrain

处理一次数据集即可：python pre_prepare_data.py --work_dir '../experiment/main/ICAE_1.1B_UPL'
使用LLama3.1需要更新transformers版本: pip install --upgrade transformers
训练模型：python ./pre_trainer.py --work_dir '../experiment/main/ICAE_1.1B_UPL' --port 14529
测试模型：python ./pre_evaluator.py --work_dir '../experiment/main/ICAE_1.1B_UPL' --batch_size 1
```

## 微调
```
cd sft
处理一次数据集即可：python instruction_prepare_data.py --work_dir '../experiment/main/ICAE_1.1B_UPL'

微调训练：python ./instruction_trainer.py --work_dir '../experiment/main/ICAE_1.1B_UPL' --port 14525
模型测试：python ./instruction_evaluator.py --work_dir '../experiment/main/ICAE_1.1B_UPL' --batch_size 1
         python ../util/evaluate_ood.py --work_dir '../experiment/main/ICAE_1.1B_UPL'
         python ../util/evaluate_iid.py --work_dir '../experiment/main/ICAE_1.1B_UPL'

训练模型和测试结果均保存在'../experiment/main/ICAE_1.1B_UPL/output'文件夹里

```
