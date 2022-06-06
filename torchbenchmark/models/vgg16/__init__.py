from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Original train batch size 256 on 4-GPU system
    # Downscale to 64 to run on single GPU device
    # Source: https://arxiv.org/pdf/1409.1556.pdf
    def __init__(self, device=None, jit=False, fuser="", dynamic_bs=0, train_bs=4, eval_bs=4, extra_args=[]):
        super().__init__(model_name="vgg16", device=device, jit=jit, fuser=fuser,
                         train_bs=train_bs, dynamic_bs=dynamic_bs, eval_bs=eval_bs, extra_args=extra_args)
