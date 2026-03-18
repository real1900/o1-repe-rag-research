import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").to(device)

activation_cache = {}
def get_activation(name):
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        activation_cache[name] = hidden_states.detach()
    return hook

target_layer_idx = base_model.config.n_layer - 1 
target_layer = base_model.transformer.h[target_layer_idx]
hook_handle = target_layer.register_forward_hook(get_activation(f"layer_{target_layer_idx}"))

dummy_text = "This paragraph contains completely irrelevant, distracting information about a side topic." * 3
dummy_input = tokenizer(dummy_text, return_tensors="pt").input_ids.to(device)

print("Extracting vector...")
with torch.no_grad():
    _ = base_model(input_ids=dummy_input)

extracted_vector = activation_cache[f"layer_{target_layer_idx}"]
hook_handle.remove()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Baseline:")
with torch.no_grad():
    baseline_outputs = base_model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
print(baseline_text)

concept_vector = extracted_vector.mean(dim=1, keepdim=True)

def create_steering_hook(alpha_val):
    def steering_hook(module, args, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        steered_states = hidden_states - (alpha_val * concept_vector)
        if isinstance(output, tuple):
            return (steered_states,) + output[1:]
        else:
            return steered_states
    return steering_hook

alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
for alpha in alphas:
    hook_handle = target_layer.register_forward_hook(create_steering_hook(alpha))
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Alpha {alpha:>5}: {text}")
    hook_handle.remove()
