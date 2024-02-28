from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
from TextSummerization.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    # def train(self):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
    #     model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
    #     seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
    #     #loading data 
    #     dataset_samsum_pt = load_from_disk(self.config.data_path)

    #     # trainer_args = TrainingArguments(
    #     #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
    #     #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
    #     #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
    #     #     evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
    #     #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
    #     # ) 


    #     trainer_args = TrainingArguments(
    #         output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
    #         per_device_train_batch_size=1, per_device_eval_batch_size=1,
    #         weight_decay=0.01, logging_steps=10,
    #         evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    #         gradient_accumulation_steps=16
    #     ) 

    #     trainer = Trainer(model=model_pegasus, args=trainer_args,
    #               tokenizer=tokenizer, data_collator=seq2seq_data_collator,
    #               train_dataset=dataset_samsum_pt["train"], 
    #               eval_dataset=dataset_samsum_pt["validation"])
        
    #     trainer.train()

    #     ## Save model
    #     model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
    #     ## Save tokenizer
    #     tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # Loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Splitting the dataset into training and validation
        train_size = int(len(dataset_samsum_pt["train"]) * 0.10)
        indices = torch.randperm(len(dataset_samsum_pt["train"])).tolist()
        train_dataset = Subset(dataset_samsum_pt["train"], indices[:train_size])

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,  # Adjust as per your GPU memory
            gradient_accumulation_steps=8,  # Adjust as per your GPU memory
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=500,
            save_steps=1e6,
            save_total_limit=1,
            fp16=False,  # Enable mixed precision training
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=train_dataset,
            eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        # Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))