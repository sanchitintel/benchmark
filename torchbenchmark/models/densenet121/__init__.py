from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION

    # Train batch size: use the training batch in paper.
    # Source: https://arxiv.org/pdf/1608.06993.pdf
    def __init__(self, device=None, jit=False, fuser="", train_bs=64, eval_bs=64, dynamic_bs=0, extra_args=[]):
        # Temporarily disable tests because it causes CUDA OOM on CI platform
        # TODO: Re-enable these tests when better hardware is available
        if device == 'cuda':
            raise NotImplementedError('CUDA disabled due to CUDA out of memory on CI GPU')
        if device == 'cpu':
            super().__init__(model_name="densenet121", device=device, jit=jit, fuser=fuser,
                             train_bs=train_bs, eval_bs=eval_bs, dynamic_bs=dynamic_bs, extra_args=extra_args)
