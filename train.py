from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification
from transformers import TrainingArguments
from transformers import Trainer
import os
from datasets import load_dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch import nn
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from safetensors.torch import load_file
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.modeling_utils import ModuleUtilsMixin
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F
from datasets import load_from_disk


os.environ["HF_DATASETS_CACHE"]="/workspace/disk1/data/.cache"
os.environ["WANDB_CACHE_DIR"]="/workspace/disk2/data/.cache/wandb"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
os.environ["WANDB_PROJECT"] = "train_cos_avg_embed" # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints
os.environ["WANDB_WATCH"] = "all" # W&B watch

dataset = load_dataset("krishnakalyan3/mj_emo_embed_100k")

dataset = dataset.rename_column("txt_embed", "labels")
dataset = dataset.rename_column("flac", "input_values")

e =  np.array(dataset["train"]["txt_raw_embed"][0:10000])
j = np.mean(e, axis=0)
np.save('/workspace/disk1/data/j_array.npy', j)
j = np.load('/workspace/disk1/data/j_array.npy')


# input_values
split = dataset["train"].train_test_split(test_size=0.005)
print(f"train_rows: {split['train'].num_rows} test_rows {split['test'].num_rows}")

class Wav2Vec2Embeddinghead(nn.Module, ModuleUtilsMixin, PyTorchModelHubMixin):
    def __init__(self, model_name, config):
        super(Wav2Vec2Embeddinghead, self).__init__()
        self.wav2vec2 = Wav2Vec2Model(config).from_pretrained(model_name)
        self.main_input_name = "input_embedding"
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        self.regression_head = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        last_hidden_states = outputs.last_hidden_state # 1, 109, 768
        pooled_output = torch.mean(last_hidden_states, dim=1) # 1, 768
        logits = self.regression_head(pooled_output) # 1, 512
        return logits

# git clone https://huggingface.co/krishnakalyan3/zero_shot_1k_cosine_model
#model_embed = Wav2Vec2Embeddinghead("facebook/wav2vec2-base-960h")
model_id = "facebook/wav2vec2-base-960h"
configuration = Wav2Vec2Config()

processor = Wav2Vec2Processor.from_pretrained(model_id)
model_embed = Wav2Vec2Embeddinghead(model_id, configuration)


def preprocess(examples):
    audio = [x["array"] for x in examples["input_values"]]
    max_duration = 10
    inputs = processor(
        audio,
        sampling_rate=16000,
        truncation=True,
        max_length=int(16000 * max_duration),
        return_attention_mask=True,
        return_tensors="pt", 
        padding=True,
        dtype = torch.float16
    )
    return inputs

train = split["train"].cast_column("input_values", Audio(sampling_rate=16000))
val = split["test"].cast_column("input_values", Audio(sampling_rate=16000))

train_set = train.map(preprocess, batched=True , remove_columns=["txt", "index", "file"])
val_set = val.map(preprocess, batched=True, remove_columns=["txt", "index", "file"])
print(f"train {train_set}, val {val_set}")
#train_set.save_to_disk('v1/train')
#val_set.save_to_disk('v1/val')


training_args = TrainingArguments(
    f"finetuned-embed-v5",
    evaluation_strategy="steps",
    remove_unused_columns=True,
    learning_rate=5e-5,
    per_device_train_batch_size=50,
    num_train_epochs=600,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=10,
    lr_scheduler_type="cosine_with_restarts",
    do_predict=True,
    report_to="wandb",
    bf16=True,
    label_names=["labels"],
    save_strategy = "steps",
    load_best_model_at_end=True,
    save_total_limit = 3
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["input_values"])
        loss_plain = 1 - F.cosine_similarity(outputs, inputs["labels"].squeeze(1)).mean()   
        # loss = cosine_similarity_loss(yhat,  y)  - j  * 0,1 or 0,5
        j_tensor = torch.tensor(j, device=torch.device('cuda:0'))

        loss = loss_plain - j_tensor * 0.5
        return (loss, outputs) if return_outputs else loss
        

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    cos_sim = cosine_similarity(predictions, labels.squeeze(1))
    average_cos_sim = np.diag(cos_sim).mean()
    return {'cos_sim': average_cos_sim}

trainer = CustomTrainer(
    model=model_embed,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub("krishnakalyan3/cosim")

