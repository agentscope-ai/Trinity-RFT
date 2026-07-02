from tests.utils.plugins.dependencies import DEPENDENCY_VALUE, dependency_func
from trinity.common.workflows.workflow import Workflow


class MainDummyWorkflow(Workflow):
    can_repeat: bool = True

    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

    def set_repeat_times(self, repeat_times, run_id_base):
        pass

    def run(self) -> list:
        return [DEPENDENCY_VALUE, dependency_func()]
