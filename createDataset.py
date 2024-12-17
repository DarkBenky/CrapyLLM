import pandas as pd


def generateXY(df):
    chatGpt = pd.read_csv('conversations.csv')
    X = []
    Y = []
    prompts = df.get("Prompts", "").tolist()
    responses = df.get("Responses", "").tolist()

    # predict next word in response
    c = 0
    for p, r in zip(prompts, responses):
        text = f"Prompt: {p} Response: "
        words = r.split()
        for i in range(1, len(words)):
            input_sequence = text + ' '.join(words[:i])
            next_word = words[i]
            X.append(input_sequence)
            Y.append(next_word)
            # input without prompt
            X.append('Response: '+' '.join(words[:i]))
            Y.append(next_word)
        c += 1
        if c % 100 == 0:
            print("predicting next word in response: ", c/len(prompts)*100, "%")

    # predict next word in prompt
    c = 0
    for p in prompts:
        words = p.split()
        for i in range(1, len(words)):
            input_sequence = "Prompt: " + ' '.join(words[:i])
            next_word = words[i]
            X.append(input_sequence)
            Y.append(next_word)
        c += 1
        if c % 100 == 0:
            print("predicting next word in prompt: ", c/len(prompts)*100, "%")

    # predict next word in response
    c = 0
    for r in responses:
        words = r.split()
        for i in range(1, len(words)):
            input_sequence = "Response: " + ' '.join(words[:i])
            next_word = words[i]
            X.append(input_sequence)
            Y.append(next_word)
        c += 1
        if c % 100 == 0:
            print("predicting next word in response only: ", c/len(prompts)*100, "%")

    return X, Y