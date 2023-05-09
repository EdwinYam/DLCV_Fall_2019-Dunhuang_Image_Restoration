import argparse

def arg_parse():
    parser= argparse.ArgumentParser(description='DLCV hw2 for image semantic segmentation')
    # Datasets parameters
    parser.add_argument('--data_path', type=str, default='../Data_Challenge2', 
                        help="root path to data directory")
    parser.add_argument('--test_path', type=str, default='../Data_Challenge2', 
                        help="root path to test data directory")
    parser.add_argument('--workers', type=int, default=4,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--img_mean', type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument('--img_std', type=float, nargs=3, default=[0.229, 0.224, 0.225])   

    # Training parameters
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--epoch', default=50, type=int,
                        help="number of training iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                        help="number of validation iterations")
    parser.add_argument('--train_batch', default=32, type=int)
    parser.add_argument('--test_batch', default=32, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--aux', action='store_true', default=True,
                        help='Auxiliary loss')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    

    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--resume', type=str, default='',
                        help="path to the trained model")
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoints', type=str, default='checkpoints')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--seg_dir', type=str, default='seg_results')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="increase output verbosity")

    args = parser.parse_args()
    return args
    
