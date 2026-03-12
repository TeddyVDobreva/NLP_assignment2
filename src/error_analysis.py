
LABELS = {0: "negative", 1: "positive"}

def get_misclassified_examples(model, raw_split, max_items: int = 8):
    model.eval()
    errs = []
    for ex in raw_split:
        tokens = tokenize(ex["text"])
        ids = numericalize(tokens, vocab)[:MAX_LEN]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
        y = int(ex["label"])
        with torch.no_grad():
            logits = model(x, lengths)
            pred = int(logits.argmax(dim=1).item())
        if pred != y:
            snippet = ex["text"].replace("\n", " ")
            snippet = snippet[:250] + ("..." if len(snippet) > 250 else "")
            errs.append((y, pred, snippet))
        if len(errs) >= max_items:
            break
    return errs

def show_errors(name: str, errs):
    print(name)
    for i,(y,p,snip) in enumerate(errs):
        print()
        print(f"error {i+1}")
        print("true:", LABELS[y], "pred:", LABELS[p])
        print("text:", snip)
