from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    optimized_for_inference = True
    def __init__(self, device=None, jit=False, fuser="", train_bs=64, eval_bs=1):
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = models.squeezenet1_1().to(self.device)
        self.eval_model = models.squeezenet1_1().to(self.device)
        self.example_inputs = (torch.randn((train_bs, 3, 224, 224)).to(self.device),)
        self.infer_example_inputs = (torch.randn((eval_bs, 3, 224, 224)).to(self.device),)

        if self.jit:
            if fuser == "llga":
                self.model = torch.jit.trace(self.model, self.example_inputs)
                self.eval_model.eval()
                self.eval_model = torch.jit.trace(self.eval_model, self.infer_example_inputs)
                self.eval_model = torch.jit.freeze(self.eval_model)
            else:
                if hasattr(torch.jit, '_script_pdt'):
                    self.model = torch.jit._script_pdt(self.model, example_inputs=[self.example_inputs, ])
                    self.eval_model = torch.jit._script_pdt(self.eval_model)
                else:
                    self.model = torch.jit.script(self.model, example_inputs=[self.example_inputs, ])
                    self.eval_model = torch.jit.script(self.eval_model)
                # model needs to in `eval`
                # in order to be optimized for inference
                self.eval_model.eval()
                self.eval_model = torch.jit.optimize_for_inference(self.eval_model)


    def get_module(self):
        return self.model, self.example_inputs

    # Original train batch size: 512, out of memory on V100 GPU
    # Use hierarchical batching to scale down: 512 = batch_size (32) * epoch_size (16)
    # Source: https://github.com/forresti/SqueezeNet
    def __init__(self, device=None, jit=False, train_bs=32, eval_bs=16, extra_args=[]):
        super().__init__(model_name="squeezenet1_1", device=device, jit=jit,
                         train_bs=train_bs, eval_bs=eval_bs, extra_args=extra_args)
        self.epoch_size = 16
    
    # Temporarily disable training because this will cause CUDA OOM in CI
    # TODO: re-enable this test when better hardware is available
    def train(self, niter=1):
        raise NotImplementedError("Temporarily disable training test because it causes CUDA OOM on T4")
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            for _ in range(self.epoch_size):
                pred = self.model(*self.example_inputs)
                y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
                loss(pred, y).backward()
            optimizer.step()
