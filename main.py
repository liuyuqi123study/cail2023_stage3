import argparse
import os
import torch
import logging

from tools.init_tool import init_all
from tools.eval_tool import valid
from config_parser import create_config
from tools.train_tool import train
import numpy as np
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()#在这里进行参数的传递
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
    parser.add_argument('--comment', help="checkpoint file path", default=None)
    parser.add_argument('--test_file', default=None)
    parser.add_argument("--seed", default=2333)#这些都是可选的参数
    args = parser.parse_args()#可以看看它是怎么读取标签数据的

    configFilePath = args.config

    config = create_config(configFilePath)
    
    if not args.test_file is None:
        config.set("data", "test_file", args.test_file)
    set_seed(args.seed)#看看有没有测试文件

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")#在这里它为要使用的gpu进行编号
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    os.system("clear")
    config.set('distributed', 'local_rank', args.local_rank)
    if config.getboolean("distributed", "use"):
        torch.cuda.set_device(gpu_list[args.local_rank])
        torch.distributed.init_process_group(backend=config.get("distributed", "backend"))
        config.set('distributed', 'gpu_num', len(gpu_list))

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    mps=torch.backends.mps.is_available()
    logger.info("mps available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    parameters = init_all(config, gpu_list,'/Users/yuqi/Downloads/cail2023第三阶段/第三阶段/2.pkl', "test", local_rank = args.local_rank)
    #在这里初始化模型
    do_test = False
    if args.do_test:
        do_test = True

    print(args.comment)#在这里开始训练
    #with torch.autograd.set_detect_anomaly(False):
    valid(parameters["model"],parameters["test_dataset"],config,gpu_list, "test")
