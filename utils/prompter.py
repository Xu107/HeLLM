"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union, List
import torch

class Prompter(object):

    def __init__(self, tokenizer,task_type: str,template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.tokenizer=tokenizer
        self.task_type=task_type
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self
    ) -> List[str]:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if self.task_type == 'general':
            instruction = "Given the user ID and purchase history, predict the most suitable item for the user."
            # instruction="You are a recommender system that can predict the next item based on historical user behavior data." \
                        # "Given the user's features and the features of historically interacted items, predict the most suitable item for the user."
        elif self.task_type == 'sequential':
            instruction = "Given the user’s purchase history, predict next possible item to be purchased."
        else:
            instruction = ""
        ins = self.template["prompt_input"].format(
            instruction=instruction
        )
        res = self.template["response_split"]
        if self._verbose:
            print(ins + res)

        self.instruct_ids=torch.tensor(self.tokenizer(ins)['input_ids'])
        self.instruct_mask=torch.tensor(self.tokenizer(ins)['attention_mask'])

        self.response_ids=torch.tensor(self.tokenizer(res)['input_ids'])
        self.response_mask=torch.tensor(self.tokenizer(res)['attention_mask'])

        # self.instruct_ids=torch.tensor(self.tokenizer.encode(ins,bos=True,eos=True))
        # self.instruct_mask=torch.tensor(len(self.instruct_ids)*[1])
        #
        # self.response_ids=torch.tensor(self.tokenizer.encode(res,bos=True,eos=True))
        # self.response_mask=torch.tensor(len(self.response_ids)*[1])
        return [self.instruct_ids,self.instruct_mask,self.response_ids,self.response_mask]

