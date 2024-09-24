import argparse
import torch
from exp.exp_sup import Exp_All_Task as Exp_All_Task_SUP
import random
import numpy as np
import wandb
from utils.ddp import is_main_process, init_distributed_mode
from taskgrouping.train_taskonomy import get_losses_and_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTS supervised training')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='ALL_task',
                        help='task name')
    parser.add_argument('--is_training', type=int,
                        required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False,
                        default='Traffic', help='model id')  # Set default based on your bash script
    parser.add_argument('--model', type=str, required=False, default='TSLANet',  # UniTS/TSLANet
                        help='model name')  # Set default based on your bash script

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='data_provider/multi_task.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')

    # ddp
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')
    parser.add_argument("--memory_check", action="store_true", default=True)
    parser.add_argument("--large_model", action="store_true", default=True)

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        required=False, default=5, help='train epochs')  # Updated to 5 as per the bash script
    parser.add_argument("--prompt_tune_epoch", type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=32,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='optimizer min learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-6, help='optimizer weight decay')  # Updated to 5e-6 as per the bash script
    parser.add_argument('--layer_decay', type=float,
                        default=None, help='optimizer layer decay')
    parser.add_argument('--des', type=str, default='Exp',
                        help='exp description')  # Updated to 'Exp' as per the bash script
    parser.add_argument('--lradj', type=str,
                        default='supervised', help='adjust learning rate')
    parser.add_argument('--clip_grad', type=float, default=100, metavar='NORM',
                        help='Clip gradient norm (default: 100 as per the bash script)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='save location of model checkpoints')
    parser.add_argument('--pretrained_weight', type=str, default=None,
                        help='location of pretrained model checkpoints')
    parser.add_argument('--debug', type=str,
                        default='online', help='wandb mode (default: online)')  # Updated to 'online'
    parser.add_argument('--project_name', type=str,
                        default='supervised_learning', help='wandb project name')  # Updated to 'supervised_learning'

    # model settings
    parser.add_argument('--d_model', type=int, default=64,
                        help='dimension of model')  # Updated to 64 as per the bash script
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3,
                        help='num of encoder layers')  # Updated to 3 as per the bash script
    parser.add_argument("--share_embedding",
                        action="store_true", default=False)
    parser.add_argument("--patch_len", type=int, default=16,
                        help='patch length')  # Updated to 16 as per the bash script
    parser.add_argument("--stride", type=int, default=16,
                        help='stride')  # Updated to 16 as per the bash script
    parser.add_argument("--prompt_num", type=int, default=10,
                        help='number of prompts')  # Updated to 10 as per the bash script
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    # task related settings
    # forecasting task
    parser.add_argument('--inverse', action='store_true',
                        help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float,
                        default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float,
                        default=1.0, help='prior anomaly ratio (%)')

    # zero-shot-forecast-new-length
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument('--zero_shot_forecasting_new_length',
                        type=str, default=None, help='unify')

    args = parser.parse_args()
    init_distributed_mode(args)
    if args.fix_seed is not None:
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)

    print('Args in experiment:')
    print(args)
    exp_name = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.d_model,
        args.e_layers,
        args.des)

    if int(args.prompt_tune_epoch) != 0:
        exp_name = 'Ptune'+str(args.prompt_tune_epoch)+'_'+exp_name
        print(exp_name)

    if is_main_process():
        wandb.init(
            name=exp_name,
            # set the wandb project where this run will be logged
            project=args.project_name,
            # track hyperparameters and run metadata
            config=args,
            mode=args.debug,
        )

    # 获取loss函数和任务
    # 假设您已经定义了其他必要的参数和设置
    parser = argparse.ArgumentParser(description='UniTS Training with Taskonomy Losses')
    # 添加其他必要的参数
    args = parser.parse_args()

    # 获取loss函数和任务
    taskonomy_loss, losses, criteria, taskonomy_tasks = get_losses_and_tasks(args)

    Exp = Exp_All_Task_SUP

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.d_model,
                args.e_layers,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.d_model,
            args.e_layers,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, load_pretrain=True)
        torch.cuda.empty_cache()
