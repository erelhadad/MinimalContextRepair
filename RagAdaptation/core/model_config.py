
from models import get_hf_scorer
from models import get_hf_scorer_single_device

class ModelConfig:

    def __init__(self, **kwargs):
        self.model_id = kwargs.get('model_id')
        self.model_hf ,self.tok_hf = get_hf_scorer(model_id=self.model_id)
        # prompt - handling

