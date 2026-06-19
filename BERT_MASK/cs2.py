from transformers import BertTokenizer, BertForMaskedLM
import torch

# Select a pre-trained BERT model.
model_name = "bert-base-uncased"  # or other models

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# List of original sentences
input_texts = [
# "a girl with long hair wearing a white dress.", #Sentence 51
# "a plate of chicken with sauce and herbs.", #Sentence 52
# "a young man sitting in front of a clock.", #Sentence 53
#        "a large jetliner flying through a cloudy blue sky.", #Sentence 54
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
    "a blue truck driving down a highway with seagulls around.",
"a forest filled with lots of tall trees.",
    "a bald eagle flying over a body of water.",
"a close up of some white flowers on a tree.",
    "a beach with palm trees and clear blue water."
]

# Iterate through each sentence
for input_text in input_texts:
    # Ensure periods are separated from words so they don't get masked together
    input_text = input_text.replace(".", " .")
    # Split the sentence into words
    tokens = input_text.split()

    # Mask and output prediction for one word at a time
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

        print(f"Prediction after masking '{tokens[i]}' is: {predicted_token_1}")
    print("\n")  # Print a blank line after outputting the prediction results of a sentence
