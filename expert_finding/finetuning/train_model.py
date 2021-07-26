import os
import expert_finding.finetuning.train as trainer

current_dir = os.path.dirname(os.path.abspath(__file__))

work_dir = os.path.join(current_dir, "output/data")
communities_path = os.path.join(work_dir, "triples")

import argparse

import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='pretrain_model')
    parser.add_argument('--triplets', help='training datasets triplets')
    parser.add_argument('--text', help='abstract for each ')
    parser.add_argument('--save_dir', help='model to save')
    parser.add_argument('--cuda-device', default=0)
    parser.add_argument('--batch-size', default=64)
    parser.add_argument('--epoch', default=4)

    args = parser.parse_args()
    print(args.triplets)
    trainer.train(work_dir)

if __name__ == '__main__':
    main()
