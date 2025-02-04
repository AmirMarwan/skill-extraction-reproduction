
import os
if not os.getcwd().lower().endswith('res3'):
    os.chdir('Res3')

import openai
import re
from utils.lib import load_api_key
from time import time#, sleep

RAW_DATA_FOLDER = 'resources/raw/'
RAW_DATA_FILES = [RAW_DATA_FOLDER+f for f in os.listdir(RAW_DATA_FOLDER) if f.endswith('.csv')]

INSTRUCTIONS = '''[Role]
You role is to create a set of rules for attracting costumer by extracting these rules from the raw data.
The rules are based on the current and previous camera information and context, and the time delta between them.
Create the appropriate rules by alternating between Japanese utterance an 'no utterance is needed'.

[Rules]
Reduce the number of rows by summarizing the raw data into a much smaller set of rows.
Do not get rid of rows containing "no customers visible".

[Input]
The raw data is formatted in a csv with the columns:
```Time delta,[Camera],[Context]
10.5,Camera info,適切な発言```

[Output Formatting]
Summarize the raw data into a much smaller set of rows with the same csv formatting without headers or backquote characters.:
10.5,"Camera info","適切な発言"
'''

def set_openai_client():
    api_keys = load_api_key()
    client = openai.OpenAI(api_key=api_keys[0])
    return client

def extract_skill(client, file_name, output_file):
    with open(file_name, 'r') as fh:
        fh.readline()
        conversation = fh.readlines()
    conversation = [','.join(line.split(',')[2:]) for line in conversation]
    if not os.path.exists(output_file):
        with open(output_file, 'w') as fh:
            fh.write('Time delta,[Camera],[Context]\n')
    chunks = [''.join(conversation[i:i+20]) for i in range(0, len(conversation), 20)]
    start_time = time()
    for idx, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": INSTRUCTIONS},
                        {"role": "user", "content": chunk},])
        print(idx, len(chunks), time()-start_time)
        with open(output_file, 'a') as fh:
            fh.write(f'{response.choices[0].message.content}\n'.replace('\n\n', '\n'))

if __name__ == '__main__':
    client = set_openai_client()
    if not os.path.exists(RAW_DATA_FOLDER.replace('/raw', '/extracted_tmp')):
        os.makedirs(RAW_DATA_FOLDER.replace('/raw', '/extracted_tmp'))
    if not os.path.exists(RAW_DATA_FOLDER.replace('/raw', '/extracted_skill')):
        os.makedirs(RAW_DATA_FOLDER.replace('/raw', '/extracted_skill'))

    for file_name in RAW_DATA_FILES:
        output_file = re.sub(r'raw/raw_t_\d+', 'extracted_tmp/extracted_tmp_0', file_name)
        extract_skill(client, file_name, output_file)
        input('Fix data manually then press Enter...')
    
    for step in range(3):
        file_name = output_file
        output_file = re.sub(f'extracted_tmp/extracted_tmp_{step}', f'extracted_tmp/extracted_tmp_{step+1}', file_name)
        extract_skill(client, file_name, output_file)
        input('Fix data manually then press Enter...')

    file_name = output_file
    output_file = re.sub(f'extracted_tmp/extracted_tmp_{step+1}', 'extracted_skill/extracted_1', file_name)
    extract_skill(client, file_name, output_file)
    print('Do not forget to fix the data manually!')

