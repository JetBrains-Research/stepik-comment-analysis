from config import Config
import os
import torch
import torch.nn.functional as F
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, set_seed

from encoder_decoder import EncoderDecoderModel


def encode_data(data, tokenizer, max_length):
    inputs = tokenizer(data["question"], padding="max_length", truncation=True, max_length=max_length)
    outputs = tokenizer(data["answer"], padding="max_length", truncation=True, max_length=max_length)

    data["input_ids"] = inputs.input_ids
    data["attention_mask"] = inputs.attention_mask
    data["decoder_input_ids"] = outputs.input_ids
    data["decoder_attention_mask"] = outputs.attention_mask
    return data


def calculate_loss(outputs, targets, batch):
    outputs = outputs.permute(0, 2, 1)
    if outputs.shape[0] != targets.shape[0] or outputs.shape[0] * targets.shape[0] == 0:
        return None

    loss = F.cross_entropy(outputs, targets, reduction="none")
    mask = targets != 1
    loss = loss * mask
    return loss.sum() / batch


def train_model(config):
    set_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    model = EncoderDecoderModel(num_hidden_layers=config.num_hidden_layers)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    raw_datasets = DatasetDict.from_csv(
        {
            "train": os.path.join(config.input_path, "df_train.csv"),
            "val": os.path.join(config.input_path, "df_eval.csv"),
        }
    )
    raw_datasets = raw_datasets.map(lambda x: encode_data(x, tokenizer, config.max_length))
    raw_datasets.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"]
    )

    dataloaders = {x: DataLoader(raw_datasets[x], batch_size=config.batch_size, shuffle=True) for x in ["train", "val"]}

    train_iterator = trange(0, config.epochs, desc="Epoch")
    iteration = 0

    for _ in train_iterator:
        for step, batch in enumerate(tqdm(dataloaders["train"], desc="Iteration")):
            optimizer.zero_grad()

            outputs = model(batch["input_ids"].to(device))
            loss = calculate_loss(outputs["logits"], batch["decoder_input_ids"].to(device), config.batch_size)
            loss.backward()
            optimizer.step()

            if iteration % 5 == 0:
                print(f"loss: {loss}")

            iteration += 1

        print("=== validation ===")
        model.eval()

        eval_loss = 0.0
        eval_steps = 0

        for step, batch in enumerate(tqdm(dataloaders["val"], desc="Eval")):
            with torch.no_grad():
                outputs = model(batch["input_ids"].to(device))
                loss = calculate_loss(outputs["logits"], batch["decoder_input_ids"].to(device), config.batch_size)

                eval_loss += loss.item()
                eval_steps += 1

        eval_loss = eval_loss / eval_steps
        print("=== validation: loss ===", eval_loss)
        torch.save(model.state_dict(), os.path.join(config.output_path, f"encoder_decoder_model_loss_{eval_loss}.pt"))


if __name__ == "__main__":
    train_model(Config())
