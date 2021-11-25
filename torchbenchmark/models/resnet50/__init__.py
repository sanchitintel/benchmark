from torchbenchmark.util.framework.vision.model_factory import TorchVisionModel
from torchbenchmark.tasks import COMPUTER_VISION

class Model(TorchVisionModel):
    task = COMPUTER_VISION.CLASSIFICATION
    optimized_for_inference = True
    def __init__(self, device=None, jit=False, fuser=""):
        super().__init__()
        self.device = device
        self.jit = jit
        self.model = models.resnet50().to(self.device)
        self.eval_model = models.resnet50().to(self.device)
        self.example_inputs = (torch.randn((1, 3, 224, 224)).to(self.device),)

        if self.jit:
            if fuser == "llga":
                self.model = torch.jit.trace(self.model, self.example_inputs)
                self.eval_model.eval()
                self.eval_model = torch.jit.trace(self.eval_model, self.example_inputs)
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

    # vision models have another model
    # instance for inference that has
    # already been optimized for inference
    def set_eval(self):
        pass

    def train(self, niter=3):
        optimizer = optim.Adam(self.model.parameters())
        loss = torch.nn.CrossEntropyLoss()
        for _ in range(niter):
            optimizer.zero_grad()
            pred = self.model(*self.example_inputs)
            y = torch.empty(pred.shape[0], dtype=torch.long, device=self.device).random_(pred.shape[1])
            loss(pred, y).backward()
            optimizer.step()

    def eval(self, niter=1):
        model = self.eval_model
        example_inputs = self.example_inputs
        example_inputs = example_inputs[0]
        for i in range(niter):
            model(example_inputs)


    def __init__(self, device=None, jit=False, train_bs=32, eval_bs=32, extra_args=[]):
        super().__init__(model_name="resnet50", device=device, jit=jit,
                         train_bs=train_bs, eval_bs=eval_bs, extra_args=extra_args)
