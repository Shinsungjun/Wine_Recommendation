'''
KAU-RML ingee hong
'''

import argparse
import numpy as np
import os
import time 
import random
import config

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F


from model import SiameseNetwork
from data.datasets import build_train_loader, build_val_loader
from utils.utils import *
# from utils.metric import *
if torch.__version__ <= '1.1.0':
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

    

def parse_args():
    ''' This is needed for torch.distributed.launch '''
    parser = argparse.ArgumentParser(description='Train instance classifier')
    # This is passed via launch.py
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--config', default=None, type=str, help='config file')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = config.get_cfg_defaults()
    cfg = merge_config(cfg, args)

    return args, cfg

def main():

    args, cfg = parse_args()

    # *  define paths ( output, logger) * #

    # get logger path
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger_path = cfg.SYS.OUTPUT_DIR+'log/'+timestamp +'.log'
    
    check_makedirs(cfg.SYS.OUTPUT_DIR)
    check_makedirs(os.path.dirname(logger_path))
    
    global logger, tWriter, vWriter
    logger = get_logger(logger_path)
    tWriter = SummaryWriter(cfg.SYS.OUTPUT_DIR+'tb_data/train')
    vWriter = SummaryWriter(cfg.SYS.OUTPUT_DIR+'tb_data/val')
    
    # * controll random seed * #
    torch.manual_seed(cfg.TRAIN.SEED)
    torch.cuda.manual_seed(cfg.TRAIN.SEED)
    torch.cuda.manual_seed_all(cfg.TRAIN.SEED) # if use multi-GPU
    np.random.seed(cfg.TRAIN.SEED)
    random.seed(cfg.TRAIN.SEED)
    cudnn.benchmark = cfg.TRAIN.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.TRAIN.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.TRAIN.CUDNN.ENABLED


    msg = '[{time}]' 'starts experiments setting '\
            '{exp_name}'.format(time = time.ctime(), exp_name = cfg.SYS.EXP_NAME)
    logger.info(msg)
    

    # * GPU env setup. * #
    distributed = False #(len(cfg.SYS.GPUS)>0) #!
    # if distributed:
    #     torch.distributed.init_process_group(
    #         backend="nccl", init_method="env://",
    #     )


    # # * define MODEL * #
    # if dist.get_rank()==0:
    #     logger.info("=> creating model ...")

    #! model build ! ! !
    model = SiameseNetwork()

    # if distributed:
    #     device = torch.device('cuda:{}'.format(args.local_rank))
    #     torch.cuda.set_device(device)

    #     model = model.to(device)
    #     model = DistributedDataParallel(
    #         model,
    #         find_unused_parameters=True,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank
    #     )
    # else:
    #     model = nn.DataParallel(model, device_ids=cfg.gpus).cuda()

    # if dist.get_rank()==0:
    #     logger.info(model)


    # * define OPTIMIZER * #
    # params_dict = dict(model.named_parameters())
    # params = [{'params': list(params_dict.values()), 'lr': cfg.TRAIN.OPT.LR}]

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.OPT.LR)
    print(cfg.TRAIN.OPT.LR)

    # * build DATALODER * #
    train_loader = build_train_loader(cfg)
    # val_loader = build_val_loader(cfg)

    # * Training setup * #
    best_a1 = 0
    best_epoch = 0

    # * RESUME * #
    if cfg.TRAIN.RESUME:
        if os.path.isfile(cfg.TRAIN.RESUME):
            if dist.get_rank()==0:
                logger.info("=> loading checkpoint '{}'".format(cfg.TRAIN.RESUME))

            checkpoint = torch.load(cfg.TRAIN.RESUME, map_location=lambda storage, loc: storage.cuda())
            
            new_epoch = ['TRAIN.START_EPOCH' ,checkpoint['epoch']]
            cfg = update_config(cfg, new_epoch)

            best_a1 = checkpoint['best_a1']
            best_epoch = checkpoint['epoch']

            # state_dict control
            checkpoint_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            checkpoint_dict = {'module.'+k: v for k, v in checkpoint_dict.items()
                            if 'module.'+k in model_dict.keys()}

            for k, _ in checkpoint_dict.items():
                logger.info('=> loading {} pretrained model {}'.format(k, cfg.TRAIN.RESUME))

            logger.info(set(model_dict)==set(checkpoint_dict))
            assert set(model_dict)==set(checkpoint_dict)

            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict, strict=True)

            # optimizer control
            optimizer.load_state_dict(checkpoint['optimizer'])

            if dist.get_rank()==0:
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAIN.RESUME, checkpoint['epoch']))
        else:
            if dist.get_rank()==0:
                logger.info("=> no checkpoint found at '{}'".format(cfg.TRAIN.RESUME))

    if args.local_rank <= 0:
        logger.info(config.summary(cfg))
    logger.info('starts training')

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.END_EPOCH+1):

        train_loss_1,train_loss_2   = train(model, train_loader, optimizer, epoch, cfg)
        print(train_loss_1, train_loss_2)
        # val_loss, val_metrics = validation(model, val_loader, cfg, epoch)

        # if args.local_rank <= 0:
        #     tWriter.add_scalar('loss', train_loss, epoch)
        #     vWriter.add_scalar('loss', val_loss, epoch)
        #     for k, v in train_metrics.items():
        #         tWriter.add_scalar('{}'.format(k), v.avg, epoch)

        #     for k, v in val_metrics.items():
        #         vWriter.add_scalar('{}'.format(k), v.avg, epoch)
                
        #     if best_a1 > val_metrics['de/abs_rel'].avg:
        #         torch.save({
        #             'epoch': epoch,
        #             'state_dict': model.module.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #         }, os.path.join(cfg.SYS.OUTPUT_DIR,'best.pth.tar'))
        #         best_a1 = val_metrics['de/abs_rel'].avg
        #         best_epoch = epoch

        #     torch.save({
        #         'epoch': epoch,
        #         'state_dict': model.module.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'best_a1': best_a1,
        #         'bset_epoch': best_epoch,
        #     }, os.path.join(cfg.SYS.OUTPUT_DIR,'checkpoint.pth.tar'))

    #     if args.local_rank <= 0:
    #         msg = 'Loss_train: {:.10f}  Loss_val: {:.10f}'.format(train_loss, val_loss)
    #         logger.info(msg)
    #         msg = 'Best a1: {}  Best Epoch: {}'.format(best_a1, best_epoch)
    #         logger.info(msg)

    # if args.local_rank <= 0:
    #     torch.save(model.module.state_dict(),
    #         os.path.join(cfg.SYS.OUTPUT_DIR, 'final_state.pth'))


def train(model, train_loader, optimizer, epoch, cfg):
    
    batch_time = AverageMeter('Batch_Time', ':6.3f')
    data_time = AverageMeter('Data_Time', ':6.3f')
    loss_meter_1 = AverageMeter('Loss', ':.4e')
    loss_meter_2 = AverageMeter('Loss', ':.4e')
    

    model.train()
    end = time.time()
    max_iter = cfg.TRAIN.END_EPOCH * len(train_loader)
    loss_fn_1 = torch.nn.BCEWithLogitsLoss()
    loss_fn_2 = torch.nn.BCEWithLogitsLoss()
    
    print(len(train_loader))
    # before = None
    for i_iter, (inputs) in enumerate(train_loader):
        # print(inputs[0] == inputs[1])
        # print(inputs[0] == inputs[2])
        

        optimizer.zero_grad()
        
        data_time.update(time.time() - end)
        


        outputs_1 = model(inputs[0],inputs[1]) #same # returns loss and predictions at each GPU
        loss_1 = loss_fn_1(outputs_1, torch.ones(outputs_1.shape))
        # l2 sigmoid 1 
        
        # model.zero_grad()
        loss_1.backward() # distributed.datapaprallel automatically gather and syncronize losses.
        optimizer.step()
        
        
        
        # loss_2
        outputs_2 = model(inputs[0],inputs[2]) #diff # returns loss and predictions at each GPU
        
        
        # print(outputs_1, outputs_2)
        
        loss_2 = loss_fn_2(outputs_2, torch.zeros(outputs_2.shape))
        
        # loss_total = loss_1 + loss_2
        # loss_total.backward()
        optimizer.zero_grad()
        
        loss_2.backward()
        optimizer.step()
        # print(loss_1.item(),loss_2.item())
        # print(loss_total.item())

    
        # *  this tis for recordings.
        n = outputs_2.shape[0] # n = batch size of each GPU #!
        if dist.is_initialized():
            loss = loss * n
            dist.all_reduce(loss)
            loss = loss / cfg.TRAIN.BATCH_SIZE
        loss_meter_1.update(loss_1.item(), cfg.TRAIN.BATCH_SIZE)
        loss_meter_2.update(loss_2.item(), cfg.TRAIN.BATCH_SIZE)
        

        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = (epoch-1)*len(train_loader) + i_iter+1

        # lr = adjust_learning_rate(optimizer,
        #                           cfg.TRAIN.OPT.LR,
        #                           max_iter,
        #                           current_iter)

        # * compute remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        
        if (i_iter+1) % cfg.TRAIN.PRINT_FREQ == 0 :#and dist.get_rank() == 0:
            msg ='Epoch: [{}/{}]({:.2f}%) [{}/{}] '\
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '\
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '\
                    'Remain {remain_time} '\
                    'Loss1 {loss_meter_1.val:.4f} '\
                    'Loss2 {loss_meter_2.val:.4f} '\
                    'lr {lr}.\n'.format(epoch, cfg.TRAIN.END_EPOCH, 
                                    # ((epoch)/cfg.TRAIN.END_EPOCH)*100, 
                                    (current_iter/max_iter)*100, 
                                    i_iter+1, len(train_loader),
                                    batch_time=batch_time,
                                    data_time=data_time,
                                    remain_time=remain_time,
                                    loss_meter_1=loss_meter_1,
                                    loss_meter_2=loss_meter_2,
                                    
                                    lr = [x['lr'] for x in optimizer.param_groups])
            logger.info(msg)
        end = time.time()
        # exit()


    return loss_meter_1.avg,loss_meter_2.avg


if __name__=='__main__':
    main()
