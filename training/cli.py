#!/usr/bin/env python3

import argparse
import os
import random

import numpy
import torch

import models
from training import data_loader, train


__doc__ = """
This script is designed as a starting point for training and fine-tuning your models using picipolka.
It includes configurable options for model loading, training arguments and parameters and model saving functionalities.

To see a full list of configurable options, use: `python cli.py --help`
"""


def cli() -> None:

    parser = argparse.ArgumentParser(description='cli tool')

    parser.add_argument('--model-name', type=str, default='vit', metavar='S', help='model name or path')
    parser.add_argument('--model-type', type=str, default='vision', metavar='S', help='model type e.g. nlp, vision')
    parser.add_argument('--tokenizer-name', type=str, default=None, metavar='S', help='tokenizer name or path')
    parser.add_argument('--dataset-name', type=str, default='mnist', metavar='S', help='dataset name(s)')
    parser.add_argument('--project-name', type=str, default=None, metavar='S', help='project name')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=None, help='-')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N', help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--num-workers', type=int, default=None, help='the number of CPUs workers')
    parser.add_argument('--use-cuda', type=lambda x: x.lower()=='true', default=True, metavar='S', help='gpu use')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='S', help='gpu id')
    parser.add_argument('--save-model', action='store_true', default=False, help='for saving the current model')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--help', type=int, default=10, metavar='N', help='see a full list of configurable options')

    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    use_mps = not args.use_mps and torch.backends.mps.is_available()

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if use_cuda:
        device = torch.device(f'cuda{args.cuda_id}') if args.cuda_id else torch.device('cuda')
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_dataloader, test_dataloader, dataset_config = data_loader.create_image_dataloaders()

    model = models.get_model(args.model_name, dataset_config)

    model.to(device)

    if args.save_model:
        os.makedirs(f"{args.project_name}", exist_ok=True)
        torch.save(model.state_dict(), f"./{args.project_name}/{args.model_name}-model.pt")


if __name__ == '__main__':
    cli()
