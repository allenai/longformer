import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
import pytorch_lightning as pl


class CoolDataset(Dataset):

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        data = torch.tensor([1, 2, 3, 4] * 128 * 8)
        mask = torch.tensor([1, 1, 1, 1] * 128 * 8)
        return data, mask


class CoolSystem(pl.LightningModule):

    def __init__(self):
        super().__init__()
        from longformer.longformer import LongformerForMaskedLM, LongformerConfig
        self.config = LongformerConfig.from_pretrained('allenai/longformer-large-4096')
        self.config.attention_mode = 'sliding_chunks'
        # self.config.num_hidden_layers = 1
        self.config.attention_dilation = [1] * self.config.num_hidden_layers
        self.config.attention_window = [256] * self.config.num_hidden_layers
        self.model = LongformerForMaskedLM(config=self.config)
        for i, layer in enumerate(self.model.roberta.encoder.layer):
            layer.attention.self.global_tokens = 1

        # self.model = AutoModel.from_pretrained('allenai/longformer-base-4096')
        # self.model = AutoModel.from_pretrained('roberta-base')

    def to(self, *args, **kwargs):
        param_count_before_moving_to_device = len(list(self.parameters()))
        super().to(*args, **kwargs)
        self.model.tie_weights()  # a new function that the user needs to implement
        param_count_after_moving_to_device = len(list(self.parameters()))
        print('==========', param_count_before_moving_to_device, param_count_after_moving_to_device)

    def forward(self, x, y):
        # print(x.shape, self.model.roberta.encoder.layer[23].attention.self.attention_window)
        return self.model(x, attention_mask=y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y)
        loss = y_hat[0].sum()
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        loader = DataLoader(CoolDataset(), batch_size=1, num_workers=0)
        return loader


if __name__ == '__main__':
    model = CoolSystem()
    trainer = pl.Trainer(num_tpu_cores=1, progress_bar_refresh_rate=10, max_epochs=10, num_sanity_val_steps=0, gpus=0,
                         checkpoint_callback=None)
    trainer.fit(model)
