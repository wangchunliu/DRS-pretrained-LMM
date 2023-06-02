# [Pre-Trained Language-Meaning Models for Multilingual Parsing and Generation (ACL 2023 Findings)](https://arxiv.org/abs/2306.00124)

## Overview

![](./figs/overview.png)

## Quick Start

#### How to use
```bash
git clone https://github.com/wangchunliu/DRS-pretrained-LMM.git
cd DRS-pretrained-LMM
```

```python
# a case of drs-text generation
from tokenization_mlm import MLMTokenizer
from transformers import MBartForConditionalGeneration

# For DRS parsing, src_lang should be set to en_XX, de_DE, it_IT, or nl_XX
tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='<drs>')
model = MBartForConditionalGeneration.from_pretrained('laihuiyuan/DRS-LMM')

# gold text: The court is adjourned until 3:00 p.m. on March 1st.
inp_ids = tokenizer.encode(
    "court.n.01 time.n.08 EQU now adjourn.v.01 Theme -2 Time -1 Finish +1 time.n.08 ClockTime 15:00 MonthOfYear 3 DayOfMonth 1",
    return_tensors="pt")

# For DRS parsing, the forced bos token here should be <drs> 
foced_ids = tokenizer.encode("en_XX", add_special_tokens=False, return_tensors="pt")
outs = model.generate(input_ids=inp_ids, forced_bos_token_id=foced_ids.item(), num_beams=5, max_length=150)
text = tokenizer.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

#### Preprocess
- Remove unsed tokens based on the training text data, and then update tokenizer and model'embeeding.
- You could download original tokenizer and model from huggingface 
[here](https://huggingface.co/facebook/mbart-large-50/tree/main), 
and download some setting files (or both updated tokenization and model)
[here](https://drive.google.com/drive/folders/1ZWZNvgjEuwU5MfxWxOyYKMi-MJ24cHWe?usp=sharing).
```bash
python update_tokenizer_model.py
```


#### Basic pre-training (B-PT)
```bash
python train_pt.py -stage bpt -max_lr 1e-4 -min_lr 1e-5 -max_len 128 -lang de_DE en_XX it_IT nl_XX -warmup_steps 3000 -decap_steps 30000
```

#### Supervised pre-training (S-PT)
```bash
python train_pt.py -stage spt -max_lr 1e-5 -min_lr 1e-5 -max_len 128 -lang de_DE en_XX it_IT nl_XX
```

#### First fine-tuning (F-FT; all data)
```bash
python train_ft.py -stage fft -max_lr 5e-5 -min_lr 1e-5 -max_len 128 -lang de_DE en_XX it_IT nl_XX -warmup_steps 3000 -decap_steps 30000
```

#### Second fine-tuning (S-FT; silver and gold data)
```bash
python train_ft.py -stage sft -max_lr 1e-5 -min_lr 1e-5 -max_len 128 -lang de_DE en_XX it_IT nl_XX
```
