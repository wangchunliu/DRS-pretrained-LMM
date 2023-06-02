# -*- coding: utf-8 -*-

import torch


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def batch_process(src, tgt, pad_id):
    """
    Concatenate two batches of data
    """
    new_src = []
    for l0, l1 in zip(src, tgt):
        len_0 = l0.ne(pad_id).sum(-1)
        len_1 = l1.ne(pad_id).sum(-1)
        seq = torch.cat([l0, torch.zeros_like(l1)], 0)
        seq[len_0:len_0 + len_1] = l1[:len_1]
        seq[len_0 + len_1:] = pad_id
        new_src.append(seq.unsqueeze(0))
    new_src = torch.cat(new_src, 0)
    max_len = max(new_src.ne(pad_id).sum(-1))

    return new_src[:, :max_len].clone()
