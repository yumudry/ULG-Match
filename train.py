import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.expression import DATASET_GETTERS
from utils.misc import AverageMeter, accuracy, f1_score, recall_score
from utils.centroidupdat import CentroidUpdater
from utils.centroidsloss import CentroidsLoss_p,SoftContrastiveLoss

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='3', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='RAF-DB', type=str,
                        choices=['cifar10', 'cifar100','RAF-DB'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=1586,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['wideresnet', 'resnext','resnet18'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=37888, type=int,#2^20 51500 17920
                        help='number of total steps to run')#Total Steps = (数据集大小 / 批大小) * Epochs
    parser.add_argument('--eval-step', default=74, type=int,#1024 206 35
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')#修改
    parser.add_argument('--batch-size', default=24, type=int, #当标签数为0.01时，修改batch_size=16 #32
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=5, type=float,#修改
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=0.001, type=float,
                        help='weight decay')#5e-4
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')#当标签数为0.01时，修改mu=14 #mu=7
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lambda-p',default=0.9, type=float, #10
                        help='part_based guaid')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,#修改!!!
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result/raf_part_centrodis/0.03/0.95/0.9',
                        help='directory to output the result')#修改resume
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=5, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--print-freq', '-p', default=20, type=int, #10
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--num-parts', default=2, type=int, #10
                    metavar='N', help='print frequency (default: 10)')


    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'resnet18':
            import models.resnet18_part as models
            model = models.build_resnet18(num_classes=args.num_classes,num_parts=args.num_parts)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == 'AgriculturalDisease':
        args.num_classes = 5 #修改类别数
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    
    elif args.dataset == 'RAF-DB':
        args.num_classes = 7 #修改类别数

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, '/home/user-lbrhk/dataset/RAF/train/')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=256,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters()], 'weight_decay': args.wdecay}
    # ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
    
    #实例化对比损失
    centroisloss = SoftContrastiveLoss()

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()

    # test_loss, test_acc_s, test_acc_g = test(args, test_loader, model, epoch=10)
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler,centroisloss)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler,centroisloss):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    prototype = CentroidUpdater(args.num_classes, feature_size=512,num_parts=args.num_parts,device=args.device)
    previous_centroids = None

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_pce = AverageMeter()
        losses_g_u = AverageMeter()
        losses_p_u = AverageMeter()
        losses_s = AverageMeter()
        mask_probs_p = AverageMeter()
        mask_probs_g = AverageMeter()

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            f_g,f_p,logits_g,logits_p = model(inputs)
     
            #全局
            logits_g = de_interleave(logits_g, 2*args.mu+1)
            logits_g_x = logits_g[:batch_size]
            logits_g_u_w, logits_g_u_s = logits_g[batch_size:].chunk(2)#平均分为俩个部分
            # print(logits_g_u_w[0,:])
            del logits_g

            #局部
            logits_p = de_interleave(logits_p,2*args.mu+1)
            logits_p_x=logits_p[:batch_size]#实际对有标签数据也取局部特征了但是我们不做处理
            logits_p_u_w,logits_p_u_s=logits_p[batch_size:].chunk(2)
            # print(logits_p_u_w[0,:,:])
            # print(logits_p_u_s[0,:,:])
            del logits_p

            #全监督局部特征损失
            loss_pce = 0
            for part in range(args.num_parts*2):
                loss_pce +=F.cross_entropy(logits_p_x[:, :, part], targets_x)
            Lp = loss_pce/(args.num_parts*2)

            #全监督全局特征损失
            Lx = F.cross_entropy(logits_g_x, targets_x, reduction='mean') 

            #局部特征无标签损失
            pseudo_label_p = torch.softmax(logits_p_u_w.detach() / args.T,dim=-2)
            # print(pseudo_label_p[0,:,:])
            max_probs_p, targets_p_u = torch.max(pseudo_label_p, dim=-2)
            mask_p = max_probs_p.ge(args.threshold).float() #mask_p 是一个与 logits_p_u_w 最后一个维度相同形状的布尔掩码
            # print(mask_p[0,:])
            # print(targets_p_u[0,:])

            Lu_p = (F.cross_entropy(logits_p_u_s, targets_p_u, 
                    reduction='none') * mask_p).mean()

            #全局特征无标签损失
            ensembled_preds = pseudo_label_p * mask_p.unsqueeze(1)#将小于阈值的设为0
            # print(ensembled_preds.shape)
            ensembled_preds = torch.sum(ensembled_preds,dim=2)
            count_above_threshold = torch.sum(mask_p, dim=1).clamp(min=1)#计算大于阈值的个数，防止除以0
            ensembled_preds = ensembled_preds/ count_above_threshold.unsqueeze(1)#求平均

            pseudo_label_g = torch.softmax(logits_g_u_w.detach()/args.T, dim=-1)
            # print(pseudo_label_g.shape)
            mask_nonzero = ensembled_preds.sum(dim=1) != 0
            refined_pseudo_label_g = torch.where(mask_nonzero.unsqueeze(1), (1-args.lambda_p)*ensembled_preds + args.lambda_p*pseudo_label_g, pseudo_label_g)
            
            
            max_probs_g, targets_g_u = torch.max(refined_pseudo_label_g, dim=-1)#沿着最后一个维度进行softmax的操作
            mask_g = max_probs_g.ge(args.threshold).float()
            # print(mask_g.shape)
            # print(targets_g_u.shape)

            Lu_g = (F.cross_entropy(logits_g_u_s, targets_g_u,
                                  reduction='none') * mask_g).mean()
            
              #全局特征
            f_g = de_interleave(f_g, 2 * args.mu + 1)
            f_g_x = f_g[:batch_size]
            f_g_u_w, f_g_u_s = f_g[batch_size:].chunk(2)
            del f_g

            #局部特征
            f_p = de_interleave(f_p, 2 * args.mu + 1)
            f_p_x = f_p[:batch_size]
            f_p_u_w, f_p_u_s = f_p[batch_size:].chunk(2)
            del f_p    

            Ls = torch.tensor(0.0)
            if epoch > 4:#修改！！！！！！
                # print(previous_centroids)
                centroids = previous_centroids.detach()
                Ls = centroisloss(f_g_u_s,f_p_u_s,centroids,targets_g_u,targets_p_u,mask_g,mask_p)
             

            loss = Lx + args.lambda_u * Lu_p + args.lambda_u * Lu_g+ args.lambda_u * Lp+args.lambda_u * Ls

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            prototype.update(f_g_x, f_g_u_w, targets_x, targets_g_u, mask_g, f_p_x, f_p_u_w, targets_p_u, mask_p)  

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_pce.update(Lp.item())
            losses_g_u.update(Lu_g.item())
            losses_p_u.update(Lu_p.item())
            losses_s.update(Ls.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs_g.update(mask_g.mean().item())
            mask_probs_p.update(mask_p.mean().item())
            
            if (batch_idx) % args.print_freq == 0:
                logger.info("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_pce: {loss_pce:.4f}. Loss_g_u: {loss_g_u:.4f}. Loss_p_u: {loss_p_u:.4f}. Loss_s: {loss_s:.4f}. Mask_g: {mask_g:.2f}. Mask_p: {mask_p:.2f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_pce=losses_pce.avg,
                        loss_g_u=losses_g_u.avg,
                        loss_p_u=losses_p_u.avg,
                        loss_s=losses_s.avg,
                        mask_g=mask_probs_g.avg,
                        mask_p=mask_probs_p.avg))
         

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, precision,recall, f1 = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_pce', losses_pce.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_g_u', losses_g_u.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_p_u', losses_p_u.avg, epoch)
            args.writer.add_scalar('train/6.mask_g', mask_probs_g.avg, epoch)
            args.writer.add_scalar('train/7.mask_p', mask_probs_p.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))
        
        prototype.compute_centroids()   
        previous_centroids = prototype.centroids.clone()  # 保存新计算的类别中心供下一个epoch使用
        prototype.reset()   

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])

            # 计算精度、召回率和F1分数
            precision, recall, f1 = f1_score(outputs, targets)
            precision_meter.update(precision.item(), inputs.shape[0])
            recall_meter.update(recall.item(), inputs.shape[0])
            f1_meter.update(f1.item(), inputs.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. Precision: {precision:.2f}. Recall: {recall:.2f}. F1: {f1:.2f}.".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    precision=precision_meter.avg,
                    recall=recall_meter.avg,
                    f1=f1_meter.avg
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    logger.info("Precision: {:.2f}".format(precision_meter.avg))
    logger.info("Recall: {:.2f}".format(recall_meter.avg))
    logger.info("F1 Score: {:.2f}".format(f1_meter.avg))
    return losses.avg, top1.avg, precision_meter.avg, recall_meter.avg, f1_meter.avg


if __name__ == '__main__':
    main()
