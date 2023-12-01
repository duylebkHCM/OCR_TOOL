import json
from pathlib import Path
from collections import defaultdict

update_card_names = {
    'MOVITEL_BACK1_GD2': 'CCCD_BACK_TYPE1',
    'MOVITEL_BACK2_GD3': 'CCCD_BACK_TYPE2',
    'MOVITEL_BAUCU1_GD3': 'VOTECARD_TYPE1',
    'MOVITEL_BAUCU2_GD3': 'VOTERCARD_TYPE2',
    'MOVITEL_BLX1_GD3': 'BLX_TYPE1',
    'MOVITEL_BLX2_GD3': 'BLX_TYPE_2',
    'MOVITEL_FRONT1_GD3':'CCCD_FRONT_TYPE1',
    'MOVITEL_FRONT2_GĐ3': 'CCCD_FRONT_TYPE1',
    'MOVITEL_PASSPORT1_GD3': 'PASSPORT_TYPE_1'
}

raw_data_path = '/media/duyla4/DATA/dataset/PROJECT_DATASET/EKYC/MOVITEL/CARD/PHASE5/raw_data'
textline_path = '/media/duyla4/DATA/dataset/PROJECT_DATASET/EKYC/MOVITEL/CARD/PHASE5/textlines'
logging_path = '/media/duyla4/DATA/dataset/PROJECT_DATASET/EKYC/MOVITEL/CARD/PHASE5/data_processing/update_json_data'
logging = open(Path(logging_path) / 'log_change_json_phase5.txt', 'w')

for card_dir in Path(raw_data_path).glob('*'):
    if card_dir.name in update_card_names:
        textline_list = list(Path(textline_path).joinpath(update_card_names[card_dir.name]).rglob('*.txt'))
                
        textline_dict = defaultdict(list)
        for path_ in textline_list:
            textline_dict[path_.stem].append({path_.parent.stem:path_.as_posix()})            

        for f_ in card_dir.rglob('*.json'):
            anno = json.load(open(f_.as_posix(), 'r', encoding='utf-8'))
            textlines_by_file = textline_dict.get(f_.stem, None)
            if textlines_by_file is None:
                print('skipping processing')
                continue
            
            change_field = []
            for box in anno['shapes']:
                if box['label'] in [list(item.keys())[0] for item in textlines_by_file]:
                    need_textline = list(filter(lambda x: list(x.keys())[0] == box['label'], textlines_by_file))
                    need_textline = need_textline[0]
                    
                    update_value = open(Path(list(need_textline.values())[0]).with_suffix('.txt'), 'r', encoding='utf-8').readline()
                    try:
                        if update_value != box['value']:
                            box['value'] = update_value
                            change_field.append(box['label'])
                    except KeyError as e:
                        print(e)
                        print('Error label', box['label'])
                        print('Error file', f_.as_posix())
                        
            if len(change_field) > 0:
                logging.write(f_.name + '\t' + '\t'.join(change_field) + '\n')
                
                with open(f_.as_posix(), 'w', encoding='utf-8') as f:
                    json.dump(anno, f, ensure_ascii=False, indent=2)    
        