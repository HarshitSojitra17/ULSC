import os
import json
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Select a pre-trained BERT model.
model_name = "bert-base-uncased"  # or other models

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# ── Load captions from i2t.py output (pipeline input) ──────────────────────
captions_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "IMAGE_TO_TEXT", "IMAGE_TO_TEXT", "captions.json"
)
if os.path.exists(captions_path):
    with open(captions_path, "r") as f:
        captions_dict = json.load(f)
    # Build ordered list: image_id -> caption string
    pipeline_texts = list(captions_dict.values())
    print(f"Loaded {len(pipeline_texts)} captions from captions.json")
    print("Captions:", pipeline_texts, "\n")
else:
    print("captions.json not found — using hardcoded list")
    pipeline_texts = None

# List of original sentences (paper's original hardcoded captions, kept as reference)
hardcoded_texts = [
# "a girl with long hair wearing a white dress.", # Sentence 51
# "a plate of chicken with sauce and herbs.", # Sentence 52
# "a young man sitting in front of a clock.", # Sentence 53
#        "a large jetliner flying through a cloudy blue sky.", # Sentence 54
#     "a group of people standing in front of a building.",
#     "a group of waterfalls in the middle of a forest.",
# "a group of ducks sitting on top of a lush green field.",
#     "a group of three white rabbits sitting next to each other.",
# "a woman in a red dress sitting on a ledge.",
#     "a brown horse standing on a white background.",
# "a man and woman crossing a street with a dog.",
#     "a small rabbit is sitting in the grass.",
# "a close up of a dog with a flower on its head.",
#     "a black and white panda eating a leafy plant.",
# "a couple of ducks standing on top of a lush green field.",
#     "a couple of giraffe standing next to each other.",
# "a close up of a fox in the snow.",
#     "a red squirrel sitting on top of a rock.",
# "an elephant with tusks walking down a dirt road.",
#     "two green parrots with red heads sitting on a branch.",
# "a close up of a giraffe with trees in the background.",
#     "a couple of elephants standing next to a body of water.",
# "a panda bear walking across a grass covered field.",
#     "a red squirrel sitting on top of a tree branch.",
# "a tiger walking across a lush green forest.",
#     "two zebras standing next to each other in a field.",
# "a white polar bear standing on top of a rocky field.",
#     "a close up of a butterfly on a leaf.",
# "a mouse sitting on top of a pile of books.",
#     "a shark with its mouth open in the water.",
# "a flock of flamingos standing on top of a lush green field.",
#     "a close up of a cat with green eyes.",
# "a group of giraffe standing next to a red car.",
#     "a panda bear standing on top of a rock.",
# "a couple of ducks floating on top of a body of water.",
#     "a brown frog sitting on top of a tree branch.",
# "a group of animals that are standing in the grass.",
#     "a small duckling sitting on a rock.",
# "a large elephant standing on top of a dry grass field.",
#     "a couple of elephants walking across a dry grass field.",
# "a koala bear climbing up a tree branch.",
#     "a group of penguins playing in the water.",
# "a group of koalas sitting on top of a lush green field.",
#     "a close up of a tiger laying on the ground.",
# "a brown and white dog laying on top of a blue rug.",
    "a blue driving down a highway with seagulls flying around.",
"a forest filled with lots of tall trees.",
    "a bald eagle flying over a body of water.",
"a close up of some white flowers on a tree.",
    "a beach with palm trees and clear blue water."
]

# Use pipeline captions if available, else fall back to hardcoded
input_texts = pipeline_texts if pipeline_texts is not None else hardcoded_texts

# ── BERT Semantic Correction ────────────────────────────────────────────────
# For each caption, simulate channel frame loss (2 words/frame masked)
# then BERT predicts the missing words — this is the semantic correction step
corrected_captions = {}

# Iterate through each sentence
for idx, input_text in enumerate(input_texts):
    print(f"--- Processing caption {idx+1}: '{input_text}' ---")
    # Append a period so BERT doesn't just predict a period when the last word is masked
    if not input_text.endswith("."):
        input_text += " ."
    input_text = input_text.replace(".", " .")
    
    # Split the sentence into words
    tokens = input_text.split()
    corrected_tokens = tokens.copy()

    import random
    
    # 1. Print predictions for all possible single-word masks (for analysis)
    for i in range(len(tokens)):
        masked_tokens = tokens.copy()  # Copy the list of words in the sentence
        masked_tokens[i] = "[MASK]"  # Use [MASK] to replace the word at the current position
        masked_text = " ".join(masked_tokens)  # Recombine into a new sentence

        inputs = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        # Find the actual indices of the [MASK] tokens in the tokenized input
        mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

        if len(mask_token_index) >= 1:
            predicted_index_1 = torch.argmax(predictions[0, mask_token_index[0]]).item()
            predicted_token_1 = tokenizer.convert_ids_to_tokens([predicted_index_1])[0]
        else:
            predicted_token_1 = "[ERROR]"

        print(f"  Masked: '{tokens[i]}' -> Predicted: '{predicted_token_1}'")

    print("\n")  # Print a blank line after outputting the prediction results of a sentence

    # 2. Simulate an actual random frame loss for the final output
    drop_idx = random.randint(0, len(tokens) - 1)
    final_masked = tokens.copy()
    final_masked[drop_idx] = "[MASK]"
    
    inputs = tokenizer(" ".join(final_masked), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        preds = model(**inputs).logits
    mask_idx = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    
    if len(mask_idx) > 0:
        pred_id = torch.argmax(preds[0, mask_idx[0]]).item()
        pred_word = tokenizer.convert_ids_to_tokens([pred_id])[0].replace("##", "")
    else:
        pred_word = tokens[drop_idx] # fallback
        
    corrected_tokens[drop_idx] = pred_word
    final_corrected_text = " ".join(corrected_tokens)

    # Store the BERT-corrected caption
    corrected_captions[str(idx)] = {
        "original": input_text,
        "corrected": final_corrected_text
    }

# ── Save corrected captions for Stable Diffusion step ──────────────────────
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corrected_captions.json")
with open(output_path, "w") as f:
    json.dump(corrected_captions, f, indent=2)

print(f"\nCorrected captions saved to: {output_path}")
print(f"Total: {len(corrected_captions)} captions processed")