import os
import time
from collections import namedtuple
import torch
import evaluate
import re
import pandas as pd
import numpy as np


class CustomEvaluate:

    def __init__(self):
        self.bleu_metric = evaluate.load("sacrebleu")
        self.comet_metric = evaluate.load("comet")
        self.bert_metric = evaluate.load("bertscore")
        self.result_score = namedtuple(
            "result_score", ["weight", "bleu", "comet", "bert_score"]
        )

    def make_prompt(self, text):

        return (
            f"Translate input sentence to Korean \n### Input: {text} \n### Translated:"
        )

    def clean_text(self, text):
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"[^\w\s]", "", text)

        return text.strip()

    def clean_term(self, text):
        text = re.sub(r"[-_\/]", " ", text)

        return text.strip()

    def count_mask(self, batchs):
        mask = []
        for batch in batchs["attention_mask"]:
            count = torch.sum(batch == 0).item()
            mask.append(count)

        return mask

    def counter_terms(self, terms, text):
        if not isinstance(terms, list):
            terms = terms.split(", ")

        return sum(text.lower().count(term.lower()) for term in terms)

    def lm_generate(self, inputs, tokenizer, model):
        examples = []
        start_time = time.time()

        for input in inputs:
            prompt = self.make_prompt(input)
            examples.append(prompt)

        example_batch = tokenizer(examples, return_tensors="pt", padding=True).to(
            model.device
        )

        mask = self.count_mask(example_batch)

        with torch.cuda.amp.autocast():
            output_tokens = model.generate(
                **example_batch, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id
            )

        outputs = [
            tokenizer.decode(
                t[len(tokenizer.encode(examples[i])) + mask[i] :],
                skip_special_tokens=True,
            )
            for i, t in enumerate(output_tokens)
        ]

        end_time = time.time()
        print(f"생성에 걸린시간 {end_time - start_time} seconds")

        return outputs

    def nmt_generate(self, inputs, tokenizer, model):
        start_time = time.time()

        example_batch = tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**example_batch)

        outputs = [
            tokenizer.decode(
                t,
                skip_special_tokens=True,
            )
            for i, t in enumerate(output_tokens)
        ]
        end_time = time.time()
        print(f"생성에 걸린시간 {end_time - start_time} seconds")

        return outputs

    def evaluate_score(self, datasets, tokenizer, model, model_type="sLM"):
        # if not len(datasets["korean"]) == len(outputs):
        #     pass
        inputs = datasets["english"]
        labels = datasets["korean"]
        term_lists = datasets["terms"]

        try:
            model_type = model_type.lower()
            if model_type == "slm":
                outputs = self.lm_generate(inputs, tokenizer, model)
            elif model_type == "nmt":
                outputs = self.nmt_generate(inputs, tokenizer, model)
            else:
                raise Exception("적합한 모델 타입을 입력하십시오. ex)sLM or NMT")
        except Exception as e:
            print(e)
            print("적합한 모델 타입을 입력하십시오. ex)sLM or NMT")
        else:
            weight_list = []
            clean_inputs = []
            clean_labels = []
            clean_outputs = []

            for input, output, label, terms in zip(inputs, outputs, labels, term_lists):

                clean_inputs.append(self.clean_text(input))
                clean_labels.append(self.clean_text(label))
                clean_outputs.append(self.clean_text(output))

                clean_term = []
                for i, term in enumerate(terms):
                    clean_term.append(self.clean_term(term))

                term_input = self.clean_term(input)
                term_output = self.clean_term(output)

                input_count = self.counter_terms(clean_term, term_input)
                predic_count = self.counter_terms(clean_term, term_output)

                weight = (
                    1.0 if predic_count > input_count else predic_count / input_count
                )
                weight_list.append(weight)

            bleu_result = self.bleu_metric.compute(
                predictions=clean_outputs, references=clean_labels
            )
            comet_result = self.comet_metric.compute(
                predictions=clean_outputs, references=clean_labels, sources=clean_inputs
            )
            bert_result = self.bert_metric.compute(
                predictions=clean_outputs, references=clean_labels, lang="ko"
            )

            result = self.result_score(
                np.mean(weight_list),
                bleu_result["score"],
                comet_result["mean_score"],
                np.mean(bert_result["f1"]),
            )

            return result
