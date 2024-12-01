import json
import prettyprinter
import pandas as pd

with open('conversations.json', 'r') as file:
    json_file = json.load(file)

prompts = []
responses = []
times = []

count = 1
for conversation in json_file:
    for res in conversation['mapping'].values():
        if m := res.get('message'):
            if (response := m.get('content').get('parts')) != [''] and (response := m.get('content').get('parts')) != None: # Check if response is empty
                if count % 2 == 0:
                    responses.append(response[0])
                    times.append(m.get('create_time'))
                else:
                    prompts.append(response[0])
                # prettyprinter.pprint(response)
                count += 1


pd = pd.DataFrame({'Prompts': prompts, 'Responses': responses, 'Times': times})
pd.to_csv('conversations.csv', index=False)

