# <p align=center>`Longformer`</p>
`Longformer` is a BERT-like model for long documents.

**\*\*\*\*\* New April 27th, 2020: A PyToch implementation of the sliding window attention  \*\*\*\*\***

We added a PyTorch implementation of the sliding window attention that doesn't require the custom CUDA kernel. It is limited in functionality but more convenient to use for finetuning on downstream tasks. 

Limitations: 
- Uses 2x more memory than our custom CUDA kernel
- Only works for the no-dilation case
- Doesn't support the autoregressive case
- As a result, it is not suitable for language modeling

However: 
- No custom CUDA kernel means it works on all devices including CPU and TPU (which the CUDA kernel doesn't support)
- Supports FP16, which offsets the 2x memory increase
- Our pretrained model doesn't use dilation making this implementation a good choice for finetuning on downstream tasks

The code snippit below and the TriviaQA scripts are updated to use this new implementation.

**\*\*\*\*\* End new information \*\*\*\*\***

### How to use

1. Download pretrained model
  * [`longformer-base-4096`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz)
  * [`longformer-large-4096`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-large-4096.tar.gz)

2. Install environment and code

    Our code relies on a custom CUDA kernel, and for now it only works on GPUs and Linux. We tested our code on Ubuntu, Python 3.7, CUDA10, PyTorch 1.2.0. If it doesn't work for your environment, please create a new issue.

    ```bash
    conda create --name longformer python=3.7
    conda activate longformer
    conda install cudatoolkit=10.0
    pip install git+https://github.com/allenai/longformer.git
    ```

3. Run the model

    ```python
    import torch
    from longformer.longformer import Longformer
    from transformers import RobertaTokenizer
    # TODO: update to use slidingchunks
    model = Longformer.from_pretrained('longformer-base-4096/')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.max_len = model.config.max_position_embeddings

    SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
    SAMPLE_TEXT = f'{tokenizer.cls_token}{SAMPLE_TEXT}{tokenizer.eos_token}'
 
    input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

    model = model.cuda()  # doesn't work on CPU
    input_ids = input_ids.cuda()

    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
    attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                         # classification: the <s> token
                                         # QA: question tokenss

    output = model(input_ids, attention_mask=attention_mask)[0]
    ```


### TriviaQA

* Training scripts: `scripts/triviaqa.py`
* Pretrained large model: [`here`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/triviaqa-longformer-large.tar.gz) (replicates leaderboard results)
* Instructions: `scripts/cheatsheet.txt`


### Compiling the CUDA kernel

We already include the compiled binaries of the CUDA kernel, so most users won't need to compile it, but if you are intersted, check `scripts/cheatsheet.txt` for instructions.


### Known issues

Please check the repo [issues](https://github.com/allenai/longformer/issues) for a list of known issues that we are planning to address soon. If your issue is not discussed, please create a new one. 


### Citing

If you use `Longformer` in your research, please cite [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150).
```
@article{Beltagy2020Longformer,
  title={Longformer: The Long-Document Transformer},
  author={Iz Beltagy and Matthew E. Peters and Arman Cohan},
  journal={arXiv:2004.05150},
  year={2020},
}
```

`Longformer` is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
