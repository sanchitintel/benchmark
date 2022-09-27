from contextlib import suppress
import torch
import typing
import timm
from torchbenchmark.util.model import BenchmarkModel
from .timm_config import TimmConfig
from typing import Generator, Tuple, Optional

class TimmModel(BenchmarkModel):
    # To recognize this is a timm model
    TIMM_MODEL = True
    # These two variables should be defined by subclasses
    DEFAULT_TRAIN_BSIZE = None
    DEFAULT_EVAL_BSIZE = None
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"

    def __init__(self, model_name, test, device, jit=False, batch_size=64, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        self.model = timm.create_model(model_name, pretrained=False, scriptable=True)
        self.cfg = TimmConfig(model = self.model, device = device)
        self.example_inputs = self._gen_input(self.batch_size)

        self.model.to(
            device=self.device
        )
        if test == "train":
            self.model.train()
        elif test == "eval":
            self.model.eval()
            if self.fuser == "fuser3":
                torch.jit.enable_onednn_fusion(True)
                if self.precision == "bf16":
                    torch._C._jit_set_autocast_mode(False)
                    with torch.cpu.amp.autocast(cache_enabled=False, dtype=torch.bfloat16):
                        import torch.fx.experimental.optimization as optimization
                        self.model = optimization.fuse(self.model)
                        self.model = torch.jit.trace(self.model, [self.example_inputs])
                else:
                    self.model = torch.jit.trace(self.model, self.example_inputs)
                self.model = torch.jit.freeze(self.model)
            else:
                self.model = torch.jit.trace(self.model, [self.example_inputs])
                self.model = torch.jit.optimize_for_inference(self.model)
            # warm-up. test_bench skips it by default
            self.model(self.example_inputs)
            self.model(self.example_inputs)
            self.model(self.example_inputs)


        if device == 'cuda':
            torch.cuda.empty_cache()
        self.amp_context = suppress

    def gen_inputs(self, num_batches:int=1) -> Tuple[Generator, Optional[int]]:
        def _gen_inputs():
            while True:
                result = []
                for _i in range(num_batches):
                    result.append((self._gen_input(self.batch_size), ))
                if self.dargs.precision == "fp16":
                    result = list(map(lambda x: (x[0].half(), ), result))
                yield result
        return (_gen_inputs(), None)

    def _gen_input(self, batch_size):
        return torch.randn((batch_size,) + self.cfg.input_size, device=self.device)

    def _gen_target(self, batch_size):
        return torch.empty(
            (batch_size,) + self.cfg.target_shape,
            device=self.device, dtype=torch.long).random_(self.cfg.num_classes)

    def _step_train(self):
        self.cfg.optimizer.zero_grad()
        with self.amp_context():
            output = self.model(self.example_inputs)
        if isinstance(output, tuple):
            output = output[0]
        target = self._gen_target(output.shape[0])
        self.cfg.loss(output, target).backward()
        self.cfg.optimizer.step()

    def _step_eval(self):
        output = self.model(self.example_inputs)
        return output

    def enable_fp16_half(self):
        self.model = self.model.half()
        self.example_inputs = self.example_inputs.half()

    def enable_channels_last(self):
        self.model = self.model.to(memory_format=torch.channels_last)
        self.example_inputs = self.example_inputs.contiguous(memory_format=torch.channels_last)

    def enable_amp(self):
        self.amp_context = lambda: torch.cuda.amp.autocast(dtype=torch.float16)

    def get_module(self):
        return self.model, (self.example_inputs,)

    def train(self):
        self._step_train()

    def eval(self) -> typing.Tuple[torch.Tensor]:
        with torch.no_grad():
            with self.amp_context():
                out = self._step_eval()
        return (out, )
