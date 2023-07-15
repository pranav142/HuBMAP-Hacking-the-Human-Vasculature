import yaml
import pandas as pd
import os
from sklearn.model_selection import KFold

def load_yaml(path: str):
    with open(path, "r") as file_obj:
        yaml_content = yaml.safe_load(file_obj)

    return yaml_content

def create_df(config, train_folder: str, polygons_json: str) -> pd.DataFrame:
    train_dir = os.path.join(config["data_path"], train_folder)
    json_df = os.path.join(config["data_path"], polygons_json)

    df = pd.read_json(json_df, lines=True)
    df["path"] = train_dir + "/" + df["id"] + ".tif"
    return df

def create_folds(df: pd.DataFrame, config):
    Fold = KFold(shuffle=True, **config["folds"])

    for n, (trn_index, val_index) in enumerate(Fold.split(df)):
        df.loc[val_index, "kfold"] = int(n)
        
    df["kfold"] = df["kfold"].astype(int)
    return df


if __name__ == "__main__":
    df = create_df(train_dir="D:\\Machine_Learning\\hubmap-hacking-the-human-vasculature\\data\\train", json_path="D:\\Machine_Learning\\hubmap-hacking-the-human-vasculature\\data\\polygons.jsonl")
    print(df.head)