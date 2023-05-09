from __future__ import absolute_import
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn


def arg_parse():
    parser = argparse.ArgumentParser(
        description='image restoration')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str,
                        default='Data_Challenge2/')
    parser.add_argument('--model', type=str,
                        default='PartialConvUnet')
    parser.add_argument('--loader', type=str,
                        default='Resize')
    parser.add_argument('--validate_dir', type=str)

    # training parameters
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--epoch', default=100, type=int,
                        help="num of training iterations")
    parser.add_argument('--train_batch', default=4, type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=4, type=int,
                        help="test batch size")
    parser.add_argument('--lr', default=5e-5, type=float,
                        help="initial learning rate")

    # resume trained model
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--output_dir', type=str)
    # others
    parser.add_argument('--save_dir', type=str, default='save_model')
    parser.add_argument('--random_seed', type=int, default=999)

    args = parser.parse_args()

    return args


def main():
    args = arg_parse()

    # decide which model to use
    from Loss.impaintLoss import ImpaintLoss as loss

    # choose loader
    if args.loader == 'Resize':
        from Loader.resizeLoader import loader
    elif args.loader == 'Crop':
        from Loader.cropLoader import loader
    elif args.loader == 'RandomMask':
        from Loader.randomMaskLoader import loader

    from Loader.validateLoader import loader as val_loader

    gated = False

    if args.model == 'PartialConvUnet':
        from Trainer.PConvUnetTrainer import PartialConvUnetTrainer
        from Net.partialConvUnet import PartialConvUnet as model
        model = model()
        trainer = PartialConvUnetTrainer(model, args.save_dir, loss, args.lr)

    elif args.model == 'ConvUnet':
        from Trainer.ConvUnetTrainer import ConvUnetTrainer
        from Net.convUnet import ConvUnet as model
        model = model()
        trainer = ConvUnetTrainer(model, args.save_dir, loss, args.lr)

    elif args.model == 'GatedConvUnet':
        from Trainer.GatedConvUnetTrainer import GatedConvUnetTrainer
        from Net.gatedConvUnet import GatedConvUnet as model
        gated = True
        model = model()
        trainer = GatedConvUnetTrainer(model, args.save_dir, loss, args.lr)

    elif args.model == 'SC-FEGAN':
        from Trainer.SC_FEGanTrainer import SC_FEGanTrainer
        from Net.gatedConvUnet import GatedConvUnet as generator
        from Net.SN_PatchGAN import SNDiscriminator as discriminator
        from Loss.SN_GANLoss import SNDisLoss, SNGenLoss
        gated = True
        gen = generator()
        dis = discriminator()
        trainer = SC_FEGanTrainer(gen, dis, args.save_dir, loss,
                                  SNGenLoss, SNDisLoss, args.lr)
    else:
        print("not supported model...")
        exit(-1)


    if args.resume != '':
        trainer.resume_model(args.resume)

    if args.mode == 'train':
        train_loader = DataLoader(loader(mode='train', dir=args.data_dir, gated=gated),
                                  batch_size=args.train_batch,
                                  num_workers=4,
                                  shuffle=True)
        val_loader = DataLoader(val_loader(dir=args.validate_dir, gated=gated),
                                  batch_size=1,
                                  num_workers=0,
                                  shuffle=False)
        trainer.train(train_loader=train_loader,
                      val_loader=val_loader, total_epoch=args.epoch)

    else:
        trainer.resume_model(args.resume)
        val_loader = DataLoader(val_loader(dir=args.validate_dir, gated=gated),
                                batch_size=1,
                                num_workers=0,
                                shuffle=False)
        trainer.inference(val_loader, args.output_dir)


if __name__ == "__main__":
    main()
