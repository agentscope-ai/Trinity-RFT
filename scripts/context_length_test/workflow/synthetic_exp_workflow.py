import torch

from trinity.common.experience import EID, Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.workflow import SimpleWorkflow, Task


@WORKFLOWS.register_module("synthetic_workflow")
class SyntheticExpWorkflow(SimpleWorkflow):
    def reset(self, task: Task):
        self.workflow_args = task.workflow_args
        self.task = task
        self.max_model_len = self.workflow_args.get("max_model_len", 4096)
        self.prompt_len = self.workflow_args.get("prompt_len", 2048)
        self.response_len = self.max_model_len - self.prompt_len
        self.dummy_token = self.workflow_args.get("dummy_token", 1024)

    def run(self):
        return [
            Experience(
                tokens=torch.full((self.max_model_len,), self.dummy_token, dtype=torch.int32),
                logprobs=torch.ones((self.response_len,), dtype=torch.float32),
                prompt_length=self.prompt_len,
                reward=torch.tensor(0.0),
                eid=EID(),
            )
            for _ in range(self.repeat_times)
        ]
