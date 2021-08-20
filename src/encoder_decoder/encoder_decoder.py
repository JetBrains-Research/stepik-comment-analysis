from transformers import AutoModel, BertGenerationConfig, BertGenerationDecoder
from torch.nn import Module


class EncoderDecoderModel(Module):
    def __init__(self, num_hidden_layers):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        decoder_config = BertGenerationConfig(
            vocab_size=250002, num_hidden_layers=num_hidden_layers, is_decoder=True, hidden_size=768
        )
        self.decoder = BertGenerationDecoder(decoder_config)

    def forward(self, x):
        embed = self.encoder(x).last_hidden_state
        out = self.decoder(inputs_embeds=embed)
        return out
