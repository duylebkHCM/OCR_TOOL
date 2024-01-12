import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

config_path = sys.argv[1]
config = yaml.safe_load(open(config_path))
update_json_config = config["update_json_data"]

update_card_names = update_json_config["update_card_names"]
logging_path = update_json_config["logging_path"]
raw_data_path = update_json_config["raw_data_path"]
textline_path = update_json_config["textline_path"]

if not Path(logging_path).exists():
    Path(logging_path).mkdir(parents=True)

logging = open(Path(logging_path) / "log_change_json.txt", "w")
error_files = open(Path(logging_path) / "log_error_file.txt", "w")

for card_dir in Path(raw_data_path).glob("*"):
    if card_dir.name in update_card_names:
        textline_list = []
        for subdir in Path(textline_path).glob("*"):
            subfiles = list(
                subdir.joinpath(update_card_names[card_dir.name]).rglob("*.txt")
            )
            textline_list += subfiles

        textline_dict = defaultdict(list)
        for path_ in textline_list:
            textline_dict[path_.stem].append({path_.parent.stem: path_.as_posix()})

        for f_ in card_dir.rglob("*.json"):
            anno = json.load(open(f_.as_posix(), encoding="utf-8"))
            textlines_by_file = textline_dict.get(f_.stem, None)
            if textlines_by_file is None:
                print("File cannot retreive ", f_.as_posix())
                file_path = f_.as_posix()
                error_files.write(file_path + "\n")
                print("skipping processing")
                continue

            change_field = []
            for box in anno["shapes"]:
                if box["label"] in [list(item.keys())[0] for item in textlines_by_file]:
                    need_textline = list(
                        filter(
                            lambda x: list(x.keys())[0] == box["label"],
                            textlines_by_file,
                        )
                    )
                    need_textline = need_textline[0]

                    update_value = open(
                        Path(list(need_textline.values())[0]).with_suffix(".txt"),
                        encoding="utf-8",
                    ).readline()
                    try:
                        if update_value != box["value"]:
                            box["value"] = update_value
                            change_field.append(box["label"])
                    except KeyError as e:
                        print(e)
                        print("Error label", box["label"])
                        print("Error file", f_.as_posix())

            if len(change_field) > 0:
                logging.write(f_.name + "\t" + "\t".join(change_field) + "\n")

                with open(f_.as_posix(), "w", encoding="utf-8") as f:
                    json.dump(anno, f, ensure_ascii=False, indent=2)
