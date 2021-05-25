import torch
import unittest
from longformer.longformer_encoder_decoder import LongformerT5ForConditionalGeneration
from longformer.sliding_chunks import pad_to_window_size
from transformers import T5Tokenizer, T5ForConditionalGeneration


class TestT5ShortSeq(unittest.TestCase):
    def _run_test(self, INPUT_TEXT, long_model_name_or_path, base_model_name_or_path):

        tokenizer = T5Tokenizer.from_pretrained(long_model_name_or_path)
        model = LongformerT5ForConditionalGeneration.from_pretrained(long_model_name_or_path)
        model.eval()
        model.config.gradient_checkpointing = True
        base_model = T5ForConditionalGeneration.from_pretrained(base_model_name_or_path)
        base_model.eval()

        data = tokenizer([INPUT_TEXT], return_tensors="pt", padding="max_length", max_length=2048)
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        decoder_input_ids = model._shift_right(input_ids[:, :5])

        attention_mask_mixed = data["attention_mask"] * torch.randint(1, 3, data["attention_mask"].size())
        # randomly set some tokens to global, this should not change the output of a short sequence

        output = model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False,)[
            0
        ].float()
        output_mixed = model(
            input_ids, attention_mask=attention_mask_mixed, decoder_input_ids=decoder_input_ids, use_cache=False,
        )[0].float()
        expected_output = base_model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
        )[0].float()

        atol = 1e-4
        self.assertTrue(torch.allclose(output, expected_output, atol=atol))
        self.assertTrue(torch.allclose(output_mixed, expected_output, atol=atol))

    def test_outout(self):
        self._run_test(
            INPUT_TEXT="Hello world!",
            long_model_name_or_path="/net/nfs2.s2-research/haokunl/exp_files/model_artifacts/t5/longt5-small-4096",
            base_model_name_or_path="t5-small",
        )
        self._run_test(
            INPUT_TEXT="It begins with the Great Hungerer. It ends in utter darkness.",
            long_model_name_or_path="/net/nfs2.s2-research/haokunl/exp_files/model_artifacts/t5/longt5-small-4096",
            base_model_name_or_path="t5-small",
        )


if __name__ == "__main__":
    unittest.main()
