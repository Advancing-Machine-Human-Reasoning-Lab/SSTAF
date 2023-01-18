import torch
from nltk.stem import WordNetLemmatizer
from transformers import RobertaTokenizer, RobertaForMaskedLM

device = "cpu"

lemmatizer = WordNetLemmatizer()
def lemmatize(w: str) -> str:
    return " ".join([lemmatizer.lemmatize(_) for _ in w.replace("_", " ").replace("-", " ").strip().lower().split()])

def predict_word(category, prevList, N=50):
    """
    category: category of the word to predict
    prevList: list of previous words
    N: number of words to return
    """

    prompt = ("An example of Cs is the<mask>.", "Examples of Cs are the<mask>.")
    multi_token_strategy = {1: 0.6822925199648933, 2: 0.25108046545898, 3: 0.0588175887850695, 4: 0.0068151744279183225}
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)

    contextSize = 5
    
    def fillMask(sentence: str, top_k: int) -> dict[str, dict[str, float]]:
        """
        sentence: prompt to fill mask in. must have at least one <mask> in it.
        top_k: number of words to return
        """
        tokenized_input_seq_pair = tokenizer.encode_plus(sentence, max_length=256, return_token_type_ids=True, truncation=True)
        input_ids = torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0).to(device)
        attention_mask = torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0).to(device)
        masked_index = torch.nonzero(input_ids[0] == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
        values, predictions = model(input_ids, attention_mask=attention_mask, labels=None)["logits"][0, masked_index, :].softmax(dim=-1).topk(top_k)
        return {tokenizer.decode(p).replace(" ", ""): v for v, p in zip(values.tolist()[0], predictions.tolist()[0])}

    def create_prompt_autoencoder():
        """
        creates a prompt for autoencoder models based on the previous words in the prompt.
        """
        if prevList:
            sentence = prompt[1].replace("C", category).lower().replace("<mask>", tokenizer.mask_token)
            if f"the{tokenizer.mask_token}" in sentence:
                for w in prevList[max(0, len(prevList) - contextSize) :]:  # keep only the last contextSize items in the list
                    sentence = sentence.replace(f"the{tokenizer.mask_token}", f"the {w}, the{tokenizer.mask_token}")
                sentence = sentence.replace(f", the{tokenizer.mask_token}", f", and the{tokenizer.mask_token}")
            else:
                for w in prevList[max(0, len(prevList) - contextSize) :]:  # keep only the last contextSize items in the list
                    sentence = sentence.replace(tokenizer.mask_token, f"{w},{tokenizer.mask_token}")
                sentence = sentence.replace(f",{tokenizer.mask_token}", f", and{tokenizer.mask_token}")
        else:
            sentence = prompt[0].replace("C", category).lower().replace("<mask>", tokenizer.mask_token)
        return sentence
    
    sentence = create_prompt_autoencoder()

    remaining = 1
    size1, size2, size3, size4 = 0, 0, 0, 0
    results1token, results2tokens, results3tokens, results4tokens = [], [], [], []
    for k, v in fillMask(sentence, 3000).items():
        probability = multi_token_strategy[1] * v
        results1token.append((k, probability))
        remaining -= probability
        size1 += 1

    for k1, v1 in fillMask(sentence.replace(tokenizer.mask_token, tokenizer.mask_token * 2, 1), 100).items():
        probability = multi_token_strategy[2] * v1
        for k2, v2 in fillMask(sentence.replace(tokenizer.mask_token, k1 + tokenizer.mask_token, 1), 15).items():
            probability *= v2
            results2tokens.append((" ".join((k1, k2)).replace(" ##", ""), probability))
            remaining -= probability
            size2 += 1

    for k1, v1 in fillMask(sentence.replace(tokenizer.mask_token, tokenizer.mask_token * 3, 1), 20).items():
        probability = multi_token_strategy[3] * v1
        for k2, v2 in fillMask(sentence.replace(tokenizer.mask_token, k1 + tokenizer.mask_token * 2, 1), 10).items():
            probability *= v2
            for k3, v3 in fillMask(sentence.replace(tokenizer.mask_token, k1 + k2 + tokenizer.mask_token, 1), 2).items():
                probability *= v3
                results3tokens.append((" ".join((k1, k2, k3)).replace(" ##", ""), probability))
                remaining -= probability
                size3 += 1

    for k1, v1 in fillMask(sentence.replace(tokenizer.mask_token, tokenizer.mask_token * 4, 1), 10).items():
        probability = multi_token_strategy[4] * v1
        for k2, v2 in fillMask(sentence.replace(tokenizer.mask_token, k1 + tokenizer.mask_token * 3, 1), 5).items():
            probability *= v2
            for k3, v3 in fillMask(sentence.replace(tokenizer.mask_token, k1 + k2 + tokenizer.mask_token * 2, 1), 2).items():
                probability *= v3
                for k4, v4 in fillMask(sentence.replace(tokenizer.mask_token, k1 + k2 + k3 + tokenizer.mask_token, 1), 1).items():
                    probability *= v4
                    results4tokens.append((" ".join((k1, k2, k3, k4)).replace(" ##", ""), probability))
                    remaining -= probability
                    size4 += 1

    if remaining < 0:
        return ValueError("remaining is negative", remaining)

    subtractedTotal, n_w_in_decaydict, results = 1, 0, {}
    i, j, k, l = 0, 0, 0, 0
    while i < size1 or j < size2 or k < size3 or l < size4:
        w, prob = "", 0
        if i < size1:
            if j < size2:
                if k < size3:
                    if l < size4:
                        m = max(results1token[i][1], results2tokens[j][1], results3tokens[k][1], results4tokens[l][1])
                    else:
                        m = max(results1token[i][1], results2tokens[j][1], results3tokens[k][1])
                else:
                    if l < size4:
                        m = max(results1token[i][1], results2tokens[j][1], results4tokens[l][1])
                    else:
                        m = max(results1token[i][1], results2tokens[j][1])
            else:
                if k < size3:
                    if l < size4:
                        m = max(results1token[i][1], results3tokens[k][1], results4tokens[l][1])
                    else:
                        m = max(results1token[i][1], results3tokens[k][1])
                else:
                    if l < size4:
                        m = max(results1token[i][1], results4tokens[l][1])
                    else:
                        m = results1token[i][1]
        else:
            if j < size2:
                if k < size3:
                    if l < size4:
                        m = max(results2tokens[j][1], results3tokens[k][1], results4tokens[l][1])
                    else:
                        m = max(results2tokens[j][1], results3tokens[k][1])
                else:
                    if l < size4:
                        m = max(results2tokens[j][1], results4tokens[l][1])
                    else:
                        m = results2tokens[j][1]
            else:
                if k < size3:
                    if l < size4:
                        m = max(results3tokens[k][1], results4tokens[l][1])
                    else:
                        m = results3tokens[k][1]
                else:
                    m = results4tokens[l][1]
        if i < size1 and m == results1token[i][1]:
            w, prob = results1token[i]
            i += 1
        elif j < size2 and m == results2tokens[j][1]:
            w, prob = results2tokens[j]
            j += 1
        elif k < size3 and m == results3tokens[k][1]:
            w, prob = results3tokens[k]
            k += 1
        else:
            w, prob = results4tokens[l]
            l += 1

        w = lemmatize(w.strip().lower()).replace(" ", "")
        if not w.strip() or not w.isalpha():
            subtractedTotal -= prob
            continue
        if w in results and results[w] != None:
            results[w] += prob
        else:
            results[w] = prob
    results = {k : v / remaining for k,v in results.items()}
    return sorted(results.items(), key=lambda item: item[1], reverse=True)[:N]

if __name__ == "__main__":
    print(predict_word("animals", ["cat", "dog", "rabbit", "hamster", "cow"]))