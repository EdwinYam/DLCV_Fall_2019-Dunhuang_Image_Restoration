import os 
import torch
from PIL import Image

import parser
import models
import data
from mean_iou_evaluate import *


def evaluate(args, model, data_loader, save_img=False):
    ''' set model to evaluate mode '''
    model.eval()
    preds_dict = dict()
    gts_dict = dict()
    with torch.no_grad():
        for idx, (img_names, imgs, segs) in enumerate(data_loader):
            imgs = imgs.cuda()
            preds = model(imgs)
            if args.model == 'baseline':
                _, preds = torch.max(preds, dim=1)
            elif args.model == 'improved':
                _, preds = torch.max(preds[0], dim=1)
            preds = preds.cpu().numpy().squeeze()
            segs = segs.numpy().squeeze()
            for img_name, pred, seg in zip(img_names,preds,segs):
                preds_dict[img_name] = pred
                gts_dict[img_name] = seg

    gts = np.concatenate(list(gts_dict.values()))
    preds = np.concatenate(list(preds_dict.values()))

    IoUs, meanIoU = mean_iou_score(preds, gts)
    
    if args.seg_dir != '' and save_img:
        #TODO
        if not os.path.exists(args.seg_dir):
            os.makedirs(args.seg_dir)
        for img_name, pred in preds_dict.items():
            img = Image.fromarray(pred.astype('uint8'))
            img.save(os.path.join(args.seg_dir, img_name))


    return IoUs, meanIoU

def test(args, model, data_loader):
    ''' set model to evaluate mode '''
    model.eval()
    preds_dict = dict()
    with torch.no_grad():
        for idx, (img_names, imgs) in enumerate(data_loader):
            imgs = imgs.cuda()
            preds = model(imgs)
            if args.model == 'baseline':
                _, preds = torch.max(preds, dim=1)
            elif args.model == 'improved':
                _, preds = torch.max(preds[0], dim=1)
            preds = preds.cpu().numpy().squeeze()
            for img_name, pred in zip(img_names,preds):
                preds_dict[img_name] = pred

    preds = np.concatenate(list(preds_dict.values()))
    
    for img_name, pred in preds_dict.items():
        img = Image.fromarray(pred.astype('uint8'))
        img.save(os.path.join(args.seg_dir, img_name))

if __name__ == '__main__':
    args = parser.arg_parse()
    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    ''' prepare data_loader '''
    if args.verbose:
        print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.SegTestset(args),
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              shuffle=False)
    
    ''' prepare best model for visualization and evaluation '''
    model = None
    if args.model == 'baseline':
        model = models.baselineNet(args).cuda()
    elif args.model == 'improved':
        model = models.improvedNet(args).cuda()
    else:
        raise NotImplementedError
    
    best_checkpoint = torch.load('model_{}_best_pth.tar'.format(args.model))
    model.load_state_dict(best_checkpoint)                                       
    test(args, model, test_loader)
