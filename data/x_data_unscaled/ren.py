import json
import os

nation = "US"
json_path = f'../x_data_aux/{nation}/abbr2id.json'

with open(json_path, 'r') as f:
    abbr2id = json.load(f)

folder_path = nation +"/"

files = os.listdir(folder_path)
print("files", files)

def rename_files(files, folder_path, abbr2id):
    for file_name in files:
        print("file_name", file_name)
        if file_name.endswith('_label_unscaled.csv'):
            abbr = file_name[:2]
            print("abbr", abbr)

            if True:
                try:
                    new_id = str(abbr2id[abbr]).zfill(2)
                    new_file_name = f'{new_id}_{str(node_id).zfill(2)}_{abbr}_label_unscaled.csv'
                    os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
                    a = os.path.join(folder_path, file_name)
                    print("a", a)
                    print(f'Renamed: {file_name} -> {new_file_name}')
                except Exception as e:
                    print(f'Error renaming {file_name}: {str(e)}')

rename_files(files, folder_path, abbr2id)
