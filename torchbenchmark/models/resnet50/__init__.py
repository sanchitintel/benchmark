from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION

    def __init__(self, device=None, jit=False, fuser="",train_bs=32, eval_bs=32, dynamic_bs=0, extra_args=[]):
        super().__init__(model_name="resnet50", device=device, jit=jit, fuser=fuser,
                         train_bs=train_bs, eval_bs=eval_bs, dynamic_bs=dynamic_bs, extra_args=extra_args)
