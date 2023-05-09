import os
import torch

import parser
import models
import data
import test

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from test import evaluate
from utils import WarmupPolyLR

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))

if __name__=='__main__':
    args = parser.arg_parse()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    ''' setup gpu '''
    torch.cuda.set_device(args.gpu)
 
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    seed = np.random.randint(100000)
    
    if not os.path.exists(os.path.join(args.checkpoints, '{}_{}'.format(args.model, seed))):
        os.makedirs(os.path.join(args.checkpoints, '{}_{}'.format(args.model, seed)))
    
    if not os.path.exists(os.path.join(args.log_dir, 'Train_info_{}_{}'.format(args.model, seed))):
        os.makedirs(os.path.join(args.log_dir, 'Train_info_{}_{}'.format(args.model, seed)))
        
    ''' load dataset and prepare dataloader '''
    print('===> prepare dataloader ... ')
    train_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='val'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=False)
    ''' load model '''
    print('===> prepare model ... ')
    model = None
    if args.model == 'baseline':
        model = models.baselineNet(args)
    elif args.model == 'improved':
        model = models.improvedNet(args)
    else:
        raise NotImplementedError
    model.cuda()

    ''' define loss '''
    criterion = None
    if args.model == 'baseline':
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'improved':
        criterion = MixSoftmaxCrossEntropyLoss()

    params_list = list() 
    if hasattr(model, 'pretrained'):
        params_list.append({'params': model.pretrained.parameters(), 'lr': args.lr})
    if hasattr(model, 'exclusive'):
        for module in model.exclusive:
            params_list.append({'params': getattr(model, module).parameters(), 'lr': args.lr * 10})
    ''' setup optimizer '''
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params_list, 
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params_list,
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    
    scheduler = WarmupPolyLR(optimizer,
                             max_iters=171*args.epoch,
                             power=0.9,
                             warmup_factor=args.warmup_factor,
                             warmup_iters=args.warmup_iters,
                             warmup_method=args.warmup_method)


    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.log_dir, 'Train_info_{}_{}'.format(args.model, seed)))
    writers = [ SummaryWriter(os.path.join(args.log_dir, 'Train_info_{}_{}_{}'.format(args.model, seed, index))) for index in range(1,9+1) ]
    
    ''' train model '''
    print('===> start training ... ')
    iters = 0
    best_mIoU = 0
    for epoch in range(1, args.epoch + 1):
        model.train()
        for idx, (_, imgs, segs) in enumerate(train_loader):
            scheduler.step()
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1
            imgs, segs = imgs.cuda(), segs.cuda()
            
            output = model(imgs)
            
            loss = None
            if args.model == 'baseline':
                loss = criterion(output, segs)
            elif args.model == 'improved':
                loss_dict = criterion(output, segs)
                loss = sum(l for l in loss_dict.values())
            optimizer.zero_grad()           # set grad of all parameters to zero
            loss.backward()                 # compute gradient for each parameters
            optimizer.step()                # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:4f}'.format(loss.data.cpu().numpy())
            print(train_info)

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            with torch.no_grad():
                IoUs, mIoU = evaluate(args, model, val_loader)
            writer.add_scalar('val_mIoU', mIoU, epoch)
            for index, iou in enumerate(IoUs):
                writers[index].add_scalar('IoU', iou, epoch)

            print('Epoch [{}]: mean IoU: {}'.format(epoch, mIoU))
 
            ''' save best model '''
            if mIoU > best_mIoU + 1e-4:
                save_model(model, os.path.join(args.checkpoints, 
                                               '{}_{}'.format(args.model, seed), 
                                               'model_{}_best_pth.tar'.format(args.model)))
                best_mIoU = mIoU

        ''' save_model '''
        save_model(model, os.path.join(args.checkpoints, 
                                       '{}_{}'.format(args.model, seed),
                                       'model_{}_{}_pth.tar'.format(args.model, epoch)))

    ''' prepare best model for visualization '''
    best_checkpoint = torch.load(os.path.join(args.checkpoints,
                                              '{}_{}'.format(args.model, seed),
                                              'model_{}_best_pth.tar'.format(args.model)))
    model.load_state_dict(best_checkpoint)
    model.eval()
    with torch.no_grad():
        IoUs, mIoU = evaluate(args, model, val_loader, save_img=True)    
    print('Best mean IoU: {}'.format(epoch, mIoU))
