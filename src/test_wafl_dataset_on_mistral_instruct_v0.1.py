from datasets import load_dataset
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "fractalego/wafl-mistral_v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.half().cuda()


def create_correct_dataset(df):
    prompts = []
    chosen = []
    for row in df.iter(batch_size=1):
        memory = row["memory"][0].strip() if row["memory"][0] is not None else ""
        rules = row["rules"][0].strip() if row["rules"][0] is not None else ""
        prompts.append("This is the summary of the bot's knowledge: \n" + memory \
                       + "\n\nThe rules are as follows:\n" + rules \
                       + "\n\nThe conversation goes as follows:\n")
        chosen.append(row["positive_conversation"][0].strip())

    return DataFrame.from_dict({"prompt": prompts, "chosen": chosen})


def create_dataset_with_incorrect_rules(df):
    """
    Create a dataset with incorrect rules.
    The rules from the original dataset are shuffled and each row contains the wrong rules
    """
    prompts = []
    chosen = []

    rules_list = df["rules"]
    index = 0
    for row in df.iter(batch_size=1):
        memory = row["memory"][0].strip() if row["memory"][0] is not None else ""
        rules = rules_list[(index + 10) % len(rules_list)]
        prompts.append("This is the summary of the bot's knowledge: \n" + memory \
                       + "\n\nThe rules are as follows:\n" + rules \
                       + "\n\nThe conversation goes as follows:\n")
        chosen.append(row["positive_conversation"][0].strip())
        index += 1

    return DataFrame.from_dict({"prompt": prompts, "chosen": chosen})


def extract_tags(prompt: str) -> list:
    """
    Extract html tags from prompt using regex.
    Returns a unique list of tags.
    """
    import re
    tags = re.findall(r'<[^>]+>', prompt)
    return list(set(tags))


if __name__ == '__main__':
    dataset = load_dataset("fractalego/wafl-functions-dataset")
    correct_df = create_correct_dataset(dataset["test"])
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for _, row in correct_df.iterrows():
        conversation = row["chosen"]
        conversation = conversation.replace("\nBot: ", "\nbot: ")
        user_string = conversation.split("\nbot: ")[0]
        prompt = row["prompt"]
        tags_in_rules = extract_tags(prompt)
        input_ids = tokenizer(prompt + user_string + "\nbot:", return_tensors="pt")
        outputs = model.generate(input_ids["input_ids"].cuda(),
                                 max_new_tokens=1024,
                                 pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(outputs[0]).replace("<|", "").replace("|>", "").replace("<s>", "").replace("</s>", "").replace(prompt, "")
        tags_in_output = extract_tags(output)
        true_positives += len(set(tags_in_rules).intersection(tags_in_output))
        false_positives += len(set(tags_in_output).difference(tags_in_rules))
        false_negatives += len(set(tags_in_rules).difference(tags_in_output))
        pass

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {2 * precision * recall / (precision + recall)}")
