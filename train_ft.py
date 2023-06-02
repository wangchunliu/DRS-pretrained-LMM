# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
from torch.nn.utils import clip_grad_norm_
from tokenization_mlm import MLMTokenizer
from transformers import MBartForConditionalGeneration

from utils.dataset import token_mask
from utils.dataset import LMMIterator
from utils.helper import shift_tokens_right
from utils.polynomial_lr_decay import PolynomialLRDecay

device = 'cuda' if cuda.is_available() else 'cpu'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluate(model, valid_loader, tokenizer, loss_fn, step):
    """Evaluation function for model"""

    loss_list = []
    with torch.no_grad():
        model.eval()
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            decoder_input = shift_tokens_right(
                tgt, tokenizer.pad_token_id,
                model.config.decoder_start_token_id)
            outputs = model(
                src, mask,
                decoder_input_ids=decoder_input)
            loss = loss_fn(
                outputs.logits.view(-1, len(tokenizer)),
                tgt.view(-1))
            loss_list.append(loss.item())
        model.train()
    avg_loss = np.mean(loss_list)
    print('[Info] valid {:05d} | loss {:.4f}'.format(step, avg_loss))

    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-seed', default=42, type=int, help='random seed')
    parser.add_argument(
        '-stage', default='fft', type=str, help='training stage')
    parser.add_argument(
        '-max_lr', default=5e-5, type=float, help='max learning rate')
    parser.add_argument(
        '-min_lr', default=1e-5, type=float, help='mini learning rate')
    parser.add_argument(
        '-max_len', default=128, type=int, help='max length of sequence')
    parser.add_argument(
        '-acc_steps', default=8, type=int, help='accumulation_steps')
    parser.add_argument(
        '-warmup_steps', default=2, type=int, help='warmup_steps')
    parser.add_argument(
        '-decap_steps', default=2, type=int, help='max_decap_steps')
    parser.add_argument(
        '-epoch', default=30, type=int, help='force stop at 20 epochs')
    parser.add_argument(
        '-batch_size', default=32, type=int, help='mini batch size')
    parser.add_argument(
        '-patience', default=6, type=int, help='early stopping')
    parser.add_argument(
        '-eval_step', default=1000, type=int, help='evaluate every x step')
    parser.add_argument(
        '-log_step', default=100, type=int, help='print log every x step')
    parser.add_argument(
        '-lang', nargs='+', help='en_XX nl_XX it_IT de_DE', required=True)

    opt = parser.parse_args()
    print('[Info]', opt)
    torch.manual_seed(opt.seed)

    model_path = "mbart-large-50"

    model = MBartForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device).train()
    if opt.stage == 'sft':
        model_mtk = 'checkpoints/mlm_fft.chkpt'
        model.load_state_dict(torch.load(model_mtk))

    tokenizer = MLMTokenizer.from_pretrained(model_path, src_lang='en_XX')
    pad_token_id = tokenizer.pad_token_id

    # load data for training
    train_loader, valid_loader = LMMIterator(opt, pad_token_id).loader

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id)
    # label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-09, lr=opt.max_lr)
    scheduler = PolynomialLRDecay(
        optimizer, warmup_steps=opt.warmup_steps,
        max_decay_steps=opt.decap_steps,
        end_learning_rate=opt.min_lr, power=2)

    tab = 0
    step = 0
    avg_loss = 1e9
    loss_list = []
    start = time.time()

    for epoch in range(opt.epoch):
        for batch in train_loader:
            step += 1
            src, tgt = map(lambda x: x.to(device), batch)

            mask = src.ne(tokenizer.pad_token_id).long()
            decoder_input = shift_tokens_right(
                tgt, tokenizer.pad_token_id,
                model.config.decoder_start_token_id)

            outputs = model(
                src, mask,
                decoder_input_ids=decoder_input)

            loss = loss_fn(
                outputs.logits.view(-1, len(tokenizer)),
                tgt.view(-1))

            loss_list.append(loss.item())

            # accumulating gradients
            loss = loss / opt.acc_steps
            loss.backward()
            scheduler.step()

            if step % opt.acc_steps == 0:
                clip_grad_norm_(
                    model.parameters(),
                    max_norm=1, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

            if step % opt.log_step == 0:
                lr = optimizer.param_groups[0]['lr']
                print('[Info] steps {:05d} | loss {:.4f} | '
                      'lr {:.6f} | second {:.2f}'.format(step,
                      np.mean(loss_list), lr, time.time() - start))
                loss_list = []
                start = time.time()

            if ((len(train_loader) > opt.eval_step
                 and step % opt.eval_step == 0)
                    or (len(train_loader) < opt.eval_step
                        and step % len(train_loader) == 0)):
                eval_loss = evaluate(
                    model, valid_loader,
                    tokenizer, loss_fn, step)
                if avg_loss >= eval_loss:
                    dir = 'checkpoints/mlm_{}.chkpt'.format(opt.stage)
                    torch.save(model.state_dict(), dir)
                    print('[Info] The checkpoint file has been updated.')
                    avg_loss = eval_loss
                    tab = 0
                else:
                    tab += 1
                if tab == opt.patience:
                    exit()


if __name__ == "__main__":
    main()
