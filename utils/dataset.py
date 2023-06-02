# -*- coding: utf-8 -*-

import random

import torch
import torch.utils.data
from tokenization_mlm import MLMTokenizer

random.seed(42)


def token_mask(seq, seq_len, replace_prob=0.35, mask_idx=0):
    if replace_prob == 0:
        return seq

    noise = torch.rand(seq.size(), dtype=torch.float).to(seq.device)
    pos_idx = torch.arange(seq.size(1)).expand_as(seq).to(seq.device)
    token_mask = (0 < pos_idx) & (pos_idx < seq_len.unsqueeze(1) - 1)
    drop_mask = (noise < replace_prob) & token_mask

    x = seq.clone()
    x.masked_fill_(drop_mask, mask_idx)

    return x


class LMMDataset(torch.utils.data.Dataset):
    """ Seq2Seq Dataset """

    def __init__(self, src_inst, tgt_inst):
        self.src_inst = src_inst
        self.tgt_inst = tgt_inst

    def __len__(self):
        return len(self.src_inst)

    def __getitem__(self, idx):
        return self.src_inst[idx], self.tgt_inst[idx]


class LMMIterator(object):
    """ Data iterator for fine-tuning BART """

    def __init__(self, opt, pad_id):

        self.opt = opt
        self.pad_id = pad_id

        self.train_src, self.train_tgt = self.read_insts('train', opt)
        self.valid_src, self.valid_tgt = self.read_insts('valid', opt)
        print('[Info] {} insts from train set'.format(len(self.train_src)))
        print('[Info] {} insts from valid set'.format(len(self.valid_src)))

        self.loader = self.gen_loader(self.train_src, self.train_tgt,
                                      self.valid_src, self.valid_tgt)

    def read_insts(self, mode, opt):
        """
        Read instances from input file
        Args:
            mode (str): 'train' or 'valid'.
        Returns:
            src_seq: list of the lists of token ids for each source seqs.
            tgt_seq: list of the lists of token ids for each tgrget seqs.
        """

        src, tgt = [], []
        model_path = "mbart-large-50"
        for lang in opt.lang:
            src_seq, tgt_seq = [], []
            src_dir = 'data/{}_{}.0'.format(mode, lang)
            tgt_dir = 'data/{}_{}.1'.format(mode, lang)
            tokenizer_0 = MLMTokenizer.from_pretrained(
                model_path, src_lang='<drs>')
            tokenizer_1 = MLMTokenizer.from_pretrained(
                model_path, src_lang=lang)

            with open(src_dir, 'r') as f1, open(tgt_dir, 'r') as f2:
                f1 = f1.readlines()
                f2 = f2.readlines()
                for i in range(min(len(f1), len(f2))):
                    s = tokenizer_0.encode(f1[i].strip())
                    t = tokenizer_1.encode(f2[i].strip())
                    src_seq.append(s[:min(len(s) - 1, self.opt.max_len)] + s[-1:])
                    tgt_seq.append(t[:min(len(t) - 1, self.opt.max_len)] + t[-1:])

            if opt.stage == 'sft':
                ups = len(src_seq)
            else:
                ups = 100000

            if mode != 'valid' and len(tgt_seq) < ups:
                times = int(ups / len(tgt_seq)) + 1
                src_seq = (src_seq * times)[:ups]
                tgt_seq = (tgt_seq * times)[:ups]

            src.extend(src_seq)
            tgt.extend(tgt_seq)
            src.extend(tgt_seq)
            tgt.extend(src_seq)

            # data for cross-lingual supervised pre-training
            if opt.stage == 'spt' and lang != 'en_XX':
                tokenizer_0 = MLMTokenizer.from_pretrained(
                    model_path, src_lang='<drs>')
                tokenizer_1 = MLMTokenizer.from_pretrained(
                    model_path, src_lang='en_XX')
                tokenizer_2 = MLMTokenizer.from_pretrained(
                    model_path, src_lang=lang)
                src_en, tgt_en, src_lg, tgt_lg = [], [], [], []
                with open('data/en_{}/{}_en_XX.0'.format(lang[:2], mode), 'r') as f1, \
                        open('data/en_{}/{}_en_XX.1'.format(lang[:2], mode), 'r') as f2:
                    f1 = f1.readlines()
                    f2 = f2.readlines()
                    for i in range(min(len(f1), len(f2))):
                        s = tokenizer_0.encode(f1[i].strip())
                        t = tokenizer_1.encode(f2[i].strip())
                        src_en.append(s[:min(len(s) - 1, self.opt.max_len)] + s[-1:])
                        tgt_en.append(t[:min(len(t) - 1, self.opt.max_len)] + t[-1:])

                with open('data/en_{}/{}_{}.0'.format(lang[:2], mode, lang), 'r') as f1, \
                        open('data/en_{}/{}_{}.1'.format(lang[:2], mode, lang), 'r') as f2:
                    f1 = f1.readlines()
                    f2 = f2.readlines()
                    for i in range(min(len(f1), len(f2))):
                        s = tokenizer_0.encode(f1[i].strip())
                        t = tokenizer_2.encode(f2[i].strip())
                        src_lg.append(s[:min(len(s) - 1, self.opt.max_len)] + s[-1:])
                        tgt_lg.append(t[:min(len(t) - 1, self.opt.max_len)] + t[-1:])

                src.extend(src_en + src_en + tgt_en + tgt_en + src_lg + src_lg + tgt_lg + tgt_lg)
                tgt.extend(src_lg + tgt_lg + src_lg + tgt_lg + src_en + tgt_en + src_en + tgt_en)

        if opt.stage == 'bpt':
            return src, src.copy()

        return src, tgt


    def gen_loader(self, train_src, train_tgt, valid_src, valid_tgt):
        """Generate pytorch DataLoader."""

        train_loader = torch.utils.data.DataLoader(
            LMMDataset(
                src_inst=train_src,
                tgt_inst=train_tgt),
            num_workers=4,
            batch_size=self.opt.batch_size,
            collate_fn=self.paired_collate_fn,
            shuffle=True)

        valid_loader = torch.utils.data.DataLoader(
            LMMDataset(
                src_inst=valid_src,
                tgt_inst=valid_tgt),
            num_workers=4,
            batch_size=self.opt.batch_size,
            collate_fn=self.paired_collate_fn)

        return train_loader, valid_loader

    def collate_fn(self, insts):
        """Pad the instance to the max seq length in batch"""

        max_len = max(len(inst) for inst in insts)

        batch_seq = [inst + [self.pad_id] * (max_len - len(inst))
                     for inst in insts]
        batch_seq = torch.LongTensor(batch_seq)

        return batch_seq

    def paired_collate_fn(self, insts):
        src_inst, tgt_inst = list(zip(*insts))
        src_inst = self.collate_fn(src_inst)
        tgt_inst = self.collate_fn(tgt_inst)

        return src_inst, tgt_inst
