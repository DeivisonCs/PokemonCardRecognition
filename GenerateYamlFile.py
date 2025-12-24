import yaml
import os

CLASSES_TXT_PATH = "data/classes.txt"
YAML_DATA_PATH = "data.yaml"

def create_data_yaml(classes_txt_path, yaml_data_path):

    if not os.path.exists(classes_txt_path):
        print(f"classes.txt file not found! Please create a classes.txt {classes_txt_path}")
        return

    with open(classes_txt_path, 'r') as f:
        classes = []

        for line in f.readlines():
            if len(line.strip()) == 0: continue
            classes.append(line.strip())
        number_of_classes = len(classes)

    data = {
        "path": "data",
        "train": "train/images",
        "val": "validation/images",
        "nc": number_of_classes,
        "names": classes
    }

    with open(yaml_data_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    return data

data = create_data_yaml(CLASSES_TXT_PATH, YAML_DATA_PATH)

if data:
    print("File content:")
    print(data)