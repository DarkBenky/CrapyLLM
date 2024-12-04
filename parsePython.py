import os

def get_python_files_text(directory):
    python_files_text = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    python_files_text[file_path] = f.read()
    return python_files_text

def get_python_functions(text):
    functions = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            function = line.strip()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("def "):
                function += "\n" + lines[j]
                j += 1
            functions.append(function)
    return functions

if __name__ == "__main__":
    directory = os.path.dirname(os.path.abspath(__file__))
    python_files_text = get_python_files_text(directory)
    Prompts = []
    Responses = []

    c = 0
    max_c = 8
    for file_path, text in python_files_text.items():
        print(f"File: {file_path}\n")
        print("Finished functions" + str(c) + " out of " + str(len(python_files_text)))
        c += 1
        if c > max_c:
            break
        function = get_python_functions(text)
        for f in function:
            # process the function for AI
            functions_words = f.split()
            for i in range(len(functions_words)// 5, len(functions_words)):
                input_sequence = "Prompt: finish the function" + ' '.join(functions_words[:i]) + "Response: "
                next_word = functions_words[i]
                Prompts.append(input_sequence)
                Responses.append(next_word)
                # input without prompt
                Prompts.append('Response: '+' '.join(functions_words[:i]))
                Responses.append(next_word)

    # save the prompts and responses to a csv file
    import pandas as pd
    df = pd.DataFrame({'Prompts': Prompts, 'Responses': Responses})
    df.to_csv('functions.csv', index=False)

