import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')


class FilterFactualDataset:

    def __init__(self, dataset_path, model_name='gpt2', max_samples=1000):

        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_samples = max_samples

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.tokenizer = None
        self.model = None
        self.dataset = []

        # Store only factual samples
        self.factual_samples = []

    def load_model(self):

        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map='auto' if self.device.type == 'cuda' else None
        )

        self.model.eval()

        print("Model loaded.\n")

    def load_dataset(self):

        with open(self.dataset_path, 'r') as f:
            data = json.load(f)

        if self.max_samples:
            data = data[:self.max_samples]

        self.dataset = data

        print(f"Loaded {len(self.dataset)} samples.\n")

    def clear_memory(self):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def compute_log_prob(self, prompt, answer):

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False
        )

        answer_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False
        )

        full_ids = prompt_ids + answer_ids

        input_ids = torch.tensor(
            [full_ids],
            device=self.device
        )

        with torch.no_grad():
            outputs = self.model(input_ids)

        logits = outputs.logits

        log_prob = 0.0

        for i, token_id in enumerate(answer_ids):

            pos = len(prompt_ids) + i

            token_logits = logits[0, pos - 1]

            token_log_probs = torch.log_softmax(
                token_logits,
                dim=-1
            )

            log_prob += token_log_probs[token_id].item()

        return log_prob

    def run(self):

        self.load_model()
        self.load_dataset()

        factual_count = 0
        counterfactual_count = 0

        print("Filtering factual samples...\n")

        for idx, item in enumerate(tqdm(self.dataset)):

            q = item['question']
            factual_answer = item['target_true']
            counterfactual_answer = item['target_new']

            # Baseline prompt
            prompt = f"Question:{q}. Answer:"

            logp_fact = self.compute_log_prob(
                prompt,
                factual_answer
            )

            logp_cf = self.compute_log_prob(
                prompt,
                counterfactual_answer
            )

            # Keep only factual samples
            if logp_fact > logp_cf:

                factual_count += 1

                # Save the whole original item
                self.factual_samples.append(item)

            else:
                counterfactual_count += 1

            if idx % 50 == 0:
                self.clear_memory()

        print("\n==============================")
        print(f"Total Samples: {len(self.dataset)}")
        print(f"Factual Samples: {factual_count}")
        print(f"Counterfactual Samples Removed: {counterfactual_count}")
        print("==============================\n")

        # Save filtered dataset
        output_file = "filtered_factual_987.json"

        with open(output_file, 'w') as f:
            json.dump(self.factual_samples, f, indent=4)

        print(f"Saved factual dataset to: {output_file}")


if __name__ == '__main__':

    analyzer = FilterFactualDataset(
        dataset_path='../../Data/gpt2_with_questions_merged.json',
        model_name='gpt2',
        max_samples=1000
    )

    analyzer.run()