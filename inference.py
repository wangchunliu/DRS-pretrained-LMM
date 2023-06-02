# -*- coding: utf-8 -*-

import argparse

import torch
from torch import cuda
from tokenization_mlm import MLMTokenizer
from transformers import MBartForConditionalGeneration

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-stage', default='sft', type=str, help='training stage')
    parser.add_argument(
        '-direc', default=0, type=str, help='gen or drs')
    parser.add_argument(
        '-bs', default=128, type=int, help='the batch size')
    parser.add_argument(
        '-nb', default=5, type=int, help='beam search num')
    parser.add_argument(
        '-seed', default=42, type=int, help='the random seed')
    parser.add_argument(
        '-length', default=150, type=int, help='the max length')
    parser.add_argument(
        '-lang', default='en_XX', type=str, help='language name')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    model_path = "mbart-large-50"
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    model_dir = 'checkpoints/lmm_sft.chkpt'
    model.load_state_dict(torch.load(model_dir))
    model.to(device).eval()

    if opt.direc == '0':
        tokenizer = MLMTokenizer.from_pretrained(model_path, src_lang='<drs>')
        forced_id = tokenizer.encode(opt.lang, add_special_tokens=False)
    else:
        tokenizer = MLMTokenizer.from_pretrained(model_path, src_lang=opt.lang)
        forced_id = tokenizer.encode('<drs>', add_special_tokens=False)

    src_seq = []
    inp_dir = 'data/{}/gold/test.{}'.format(opt.lang[:2], opt.direc)
    with open(inp_dir, 'r') as fin:
        for line in fin.readlines():
            src_seq.append(line.strip())

    with open('./data/outputs/lmm_{}.{}'.format(
            opt.lang, opt.direc), 'w') as fout:
        for idx in range(0, len(src_seq), opt.bs):
            inp = tokenizer.batch_encode_plus(
                src_seq[idx: idx + opt.bs],
                padding=True, return_tensors='pt')
            src = inp['input_ids'].to(device)
            mask = inp['attention_mask'].to(device)
            outs = model.generate(
                input_ids=src,
                attention_mask=mask,
                num_beams=opt.nb,
                max_length=opt.length,
                forced_bos_token_id=forced_id)
            for x, y in zip(outs, src_seq[idx:idx + opt.bs]):
                text = tokenizer.decode(
                    x.tolist(), skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                if len(text.strip()) == 0:
                    text = y
                fout.write(text.strip() + '\n')


if __name__ == "__main__":
    main()
