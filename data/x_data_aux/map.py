import json
import os

import pandas as pd

csv_file = '../data/x_data_aux/{nation}/statemappingUS.csv'
df = pd.read_csv(csv_file)

id2state = {row['id']: row['State'] for _, row in df.iterrows()}
state2id = {row['State']: row['id'] for _, row in df.iterrows()}
id2abbr = {row['id']: row['Abbr'] for _, row in df.iterrows()}
abbr2id = {row['Abbr']: row['id'] for _, row in df.iterrows()}
state2abbr = {row['State']: row['Abbr'] for _, row in df.iterrows()}
abbr2state = {row['Abbr']: row['State'] for _, row in df.iterrows()}

combinations = [('id2state', id2state), ('state2id', state2id), ('id2abbr', id2abbr), ('abbr2id', abbr2id), ('state2abbr', state2abbr), ('abbr2state', abbr2state)]

output_dir = '../data/x_data_aux/US/'
os.makedirs(output_dir, exist_ok=True)

for name, data in combinations:
    json_file = os.path.join(output_dir, f'{name}.json')
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Data has been written to {json_file}')
