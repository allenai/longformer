import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
import pytorch_lightning as pl
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class CoolDataset(Dataset):

    def __init__(self, seq_len, global_tokens, **kwargs):
        self.seq_len = seq_len
        self.global_tokens = global_tokens
        super().__init__(**kwargs)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        data = torch.tensor([1, 2, 3, 4] * (self.seq_len // 4))
        mask = torch.tensor([1, 1, 1, 1] * (self.seq_len // 4))
        mask[:10] = 2
        mask[-10:] = 0
        return data, mask


class CoolSystem(pl.LightningModule):

    def __init__(self, args, attention_mode='sliding_chunks', attention_window=256, batch_size=1, seq_len=4096, global_tokens=0):
        super().__init__()

        # self.config = LongformerConfig.from_pretrained('allenai/longformer-large-4096')
        # self.config.attention_mode = attention_mode
        # # self.config.num_hidden_layers = 1
        # self.config.attention_dilation = [1] * self.config.num_hidden_layers
        # self.config.attention_window = [attention_window] * self.config.num_hidden_layers
        # self.model = LongformerForMaskedLM(config=self.config)
        # for i, layer in enumerate(self.model.roberta.encoder.layer):
        #     layer.attention.self.global_tokens = 0
        #     layer.attention.self.attention_mode = attention_mode
        #     layer.attention.self.attention_window = attention_window
        # # self.model = self.model.roberta.encoder.layer[0].attention.self

        # # self.model = AutoModel.from_pretrained('allenai/longformer-base-4096')
        # # self.model = AutoModel.from_pretrained('roberta-base')
        self.batch_size = batch_size
        self.count = 0
        self.seq_len = seq_len
        self.global_tokens = global_tokens

        from longformer.longformer import LongformerForMaskedLM, LongformerConfig
        self.config = LongformerConfig.from_pretrained(args.model)
        self.config.attention_mode = args.attention_mode
        self.model = LongformerForMaskedLM.from_pretrained(args.model, config=self.config)
        for i, layer in enumerate(self.model.roberta.encoder.layer):
            layer.attention.self.global_tokens = global_tokens
            layer.attention.self.attention_window = attention_window

    def to(self, *args, **kwargs):
        param_count_before_moving_to_device = len(list(self.parameters()))
        super().to(*args, **kwargs)
        self.model.tie_weights()  # a new function that the user needs to implement
        param_count_after_moving_to_device = len(list(self.parameters()))
        print('==========', param_count_before_moving_to_device, param_count_after_moving_to_device)

    def forward(self, x, y):
        print(x.shape, self.model.roberta.encoder.layer[11].attention.self.attention_window,
              self.model.roberta.encoder.layer[11].attention.self.global_tokens,
              self.model.roberta.encoder.layer[11].attention.self.attention_mode)
        return self.model(x, attention_mask=y)
        # return self.model(x[:, :, None].expand(1, 4096, 768).float())

    def training_step(self, batch, batch_idx):
        x, y = batch
        # if self.count == 1:
        #     exit()
        # self.count += 1
        y_hat = self(x, y)
        loss = y_hat[0].sum()
        # xm.mark_step()
        # exit()
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        loader = DataLoader(CoolDataset(seq_len=self.seq_len, global_tokens=self.global_tokens), batch_size=self.batch_size, num_workers=0)
        return loader


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--attention_window', type=int, default=256)
    parser.add_argument('--attention_mode', default='sliding_chunks')
    parser.add_argument('--seq_len', type=int, default=4096)
    parser.add_argument('--tpus', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--global_tokens', type=int, default=0)
    parser.add_argument('--model')
    args = parser.parse_args()
    model = CoolSystem(args, attention_mode=args.attention_mode, attention_window=args.attention_window, batch_size=args.batch_size, seq_len=args.seq_len, global_tokens=args.global_tokens)
    trainer = pl.Trainer(num_tpu_cores=args.tpus, progress_bar_refresh_rate=5, max_epochs=10, num_sanity_val_steps=0,
                         checkpoint_callback=None, gpus=args.gpus)
    trainer.fit(model)

if __name__ == '__main__':
    main()