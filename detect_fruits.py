import json
from pathlib import Path
from typing import Dict

import torch
import click
from tqdm import tqdm

model_torch = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)


def filter_with_conf(subject, input_dict, confidence):
    if confidence > 0:
        return len([x for x in input_dict if x['name'] == subject and x['confidence'] > confidence])
    else:
        return len([x for x in input_dict if x['name'] == subject])


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """

    # TODO: Implement detection method.
    results = model_torch(img_path)
    banana = filter_with_conf('banana', json.loads(results.pandas().xyxy[0].to_json(orient="records")), 0.75)
    apple = filter_with_conf('apple', json.loads(results.pandas().xyxy[0].to_json(orient="records")), 0)
    orange = filter_with_conf('orange', json.loads(results.pandas().xyxy[0].to_json(orient="records")), 0.75)

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
