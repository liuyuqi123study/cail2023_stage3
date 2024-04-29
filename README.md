## Introduction
This is the prediction job for CAIL2023 stage 3 contest in task of similar case retrieval.
## 数据预处理
首先您需要先运行SCR-Preprocess中的process4supervised文件
在完成数据预处理后，输出的文件在output-supervised中  
然后，再将数据移动到SCR-Experiment的input_data文件夹下。  
即input_data/query和input_data/candidates
## config的编写
我在这里修改了batc_size希望可以不那么占内存。原来的值是16  

## 关于Bert-Base-Chinese  
因为有墙的原因，我需要将bert-base-Chinese放在本地目录下
## 关于运行
最终的运行文件我写在SCR-Experiment目录下，具体的执行方法，需要在SCR-Experiment目录下，复制您目录中run_train.sh文件的绝对路径-》回车即可。最后的输出结果在SCR-Experiment/result/EDBERT/test0/prediction.json文件中。  

## 关于硬件
数据预处理部分我是用CPU进行的，在实验部分是使用mps进行的。如果您没有mps出现了报错，可以修改一下文件中的配置。我已经尽力使用了条件语句。  
## 参考
@inproceedings{yao-etal-2022-leven,
    title = "{LEVEN}: A Large-Scale {C}hinese Legal Event Detection Dataset",
    author = "Yao, Feng and Xiao, Chaojun and Wang, Xiaozhi and Liu, Zhiyuan and Hou, Lei and Tu, Cunchao and Li, Juanzi and Liu, Yun and Shen, Weixing and Sun, Maosong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    year = "2022",
    url = "https://aclanthology.org/2022.findings-acl.17",
    doi = "10.18653/v1/2022.findings-acl.17",
    pages = "183--201",
}
