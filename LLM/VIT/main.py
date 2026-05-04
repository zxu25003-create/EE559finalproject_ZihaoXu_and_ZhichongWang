import os
import torch
import argparse
import datetime
from solver import Solver


def main(args):
    os.makedirs(args.model_path,  exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    solver = Solver(args)
    solver.train()
    solver.plot_graphs()
    solver.test(split="test")


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


def update_args(args):
    args.model_path  = os.path.join(args.model_path, args.dataset)
    args.output_path = os.path.join(args.output_path, args.dataset)
    args.n_patches   = (args.image_size // args.patch_size) ** 2
    args.is_cuda     = torch.cuda.is_available()

    if args.is_cuda:
        print("Using GPU")
    else:
        print("Cuda not available. Using CPU.")

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of Vision Transformer for pneumonia classification'
    )

    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs for learning-rate warmup')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes in the dataset')
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for data loaders')
    parser.add_argument('--lr', type=float, default=5e-4, help='peak learning rate')
    parser.add_argument('--output_path', type=str, default='./outputs', help='path to store training curves and metrics')

    parser.add_argument('--dataset', type=str, default='pneumonia')
    parser.add_argument("--image_size", type=int, default=224, help='image size')
    parser.add_argument("--patch_size", type=int, default=16, help='patch size')
    parser.add_argument("--n_channels", type=int, default=3, help='number of input channels')
    parser.add_argument('--data_path', type=str, default='../../data/', help='dataset root path')

    parser.add_argument("--use_torch_transformer_layers", type=bool, default=False, help="use PyTorch TransformerEncoder layers")
    parser.add_argument("--embed_dim", type=int, default=64, help='embedding dimension')
    parser.add_argument("--n_attention_heads", type=int, default=4, help='number of attention heads')
    parser.add_argument("--forward_mul", type=int, default=2, help='MLP expansion multiplier')
    parser.add_argument("--n_layers", type=int, default=6, help='number of Transformer encoder layers')
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout rate')
    parser.add_argument('--model_path', type=str, default='./model', help='path to store trained model')
    parser.add_argument("--load_model", type=bool, default=False, help="load saved model")

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args = update_args(args)
    print_args(args)

    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
