"""Import + smoke check for the ML stack."""
import sys

import peft
import torch
import transformers

print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
x = torch.randn(8, 16)
lin = torch.nn.Linear(16, 4)
print("forward ok:", lin(x).shape)
print("threads:", torch.get_num_threads())
