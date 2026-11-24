# AI Use Disclosure — Homework 12

## How AI was used

I used an AI coding assistant (Claude Code) to help plan, write and run this assignment. It:

- broke the PDF into tasks (GPTQ comparison, distillation, the two sweeps, per-class analysis, final comparison);
- It heplped me understand the topics and concepts.
- set up the environment (a WSL Ubuntu virtual environment with PyTorch, `transformers`, `optimum` and GPT-QModel);




## Problems found and how they were fixed
1. **Broken Windows environment.** On native Windows, `transformers` failed to import because `torchaudio 2.10` did not match
   `torch 2.6`. GPT-QModel, the GPTQ backend that Hugging Face uses, also could not be built because the machine has no MSVC compiler. Instead of
   modifying the global Windows environment, both parts were run in a separate WSL2 Ubuntu venv on the same GPU.
2. **Incorrect quantised-layer label (AI error).** The first Part 1 run reported the GPTQ model's layer type as
   `nn.Linear (fp16)`. The check looked for the substring `Quant` in the class name, but GPT-QModel's Ampere kernel class is
   called `MarlinLinear`. I noticed the contradiction: the on-disk size and resident weight memory had halved, which cannot
   happen with fp16 layers. Inspecting the reloaded model showed 72 `MarlinLinear` modules holding packed `int32` `qweight` tensors.
   The check was changed to test the module path (`gptqmodel...qlinear`) and Part 1 was re-run.
3. **Peak-memory baseline.** The first Part 2 smoke test measured peak memory without subtracting tensors that were already
   on the GPU (the cached teacher logits). The benchmark now subtracts the memory allocated before the model is loaded.


