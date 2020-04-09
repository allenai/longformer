# <p align=center>`Longformer`</p>
`Longformer` is a BERT-like model for long documents.

### How to use

1. Download pretrained model
  * [`longformer-base-4096`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz)
  * [`longformer-large-4096`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-large-4096.tar.gz)

2. Install environment and code

    ```bash
    conda create --name longformer python=3.7
    conda activate longformer
    conda install cudatoolkit=10.0
    pip install git+https://github.com/allenai/longformer.git
    ```
    We tested our code on Ubuntu, Python 3.7, CUDA10, PyTorch 1.2.0. If it doesn't work for your environment, please create a new issue.

3. Run the model

    ```python
    import torch
    from longformer.longformer import Longformer
    from transformers import RobertaTokenizer

    model = Longformer.from_pretrained('longformer-base-4096/')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.max_len = model.config.max_position_embeddings

    SAMPLE_TEXT = ' '.join(['Hello world! cécé herlolip'] * 450)  # long input document
    SAMPLE_TEXT = f'{tokenizer.cls_token}{SAMPLE_TEXT}{tokenizer.eos_token}'
 
    input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

    model = model.cuda()  # doesn't work on CPU
    input_ids = input_ids.cuda()

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    attention_mask[:, [0, 1, 2, 3, 21, 513,]] =  2  # 0: no attention, 1: local attention, 2: global attention

    output = model(input_ids, attention_mask=attention_mask)[0]
    ```


### Compiling the CUDA kernel

Most users won't need to compile the CUDA kernel, but if you are intersted, check `scripts/cheatsheet.txt` for instructions.



### Citing

If you use `Longformer` in your research, please cite [Longformer: The Long-Document Transformer](https://arxiv.org/).
```
@inproceedings{Beltagy2020Longformer,
  title={Longformer: The Long-Document Transformer},
  author={Iz Beltagy and Matthew E. Peters and Arman Cohan},
  year={2020},
}
```

`Longformer` is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
