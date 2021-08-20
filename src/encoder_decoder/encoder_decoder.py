from transformers import AutoModel, BertGenerationConfig, BertGenerationDecoder
from torch.nn import Module


class EncoderDecoderModel(Module):
    def __init__(self, model_checkpoint, hidden_size=768, num_hidden_layers=2):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_checkpoint)
        decoder_config = BertGenerationConfig(
            vocab_size=250002, num_hidden_layers=num_hidden_layers, is_decoder=True, hidden_size=hidden_size
        )
        self.decoder = BertGenerationDecoder(decoder_config)

    def forward(self, x):
        embed = self.encoder(x).last_hidden_state
        out = self.decoder(inputs_embeds=embed)
        return out
