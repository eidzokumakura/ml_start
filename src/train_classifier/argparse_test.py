# python
import argparse
from typing import Any

# 3rdarty
import cv2

# project


def inference_classifier(classifier: Any, path_to_image) -> str:
    """Метод для инференса классификатора на единичном изображении

    Args:
        classifier (Any): _description_
        path_to_image (_type_): _description_

    Returns:
        str: _description_
    """
    pass


def load_classifier(
    name_of_classifier: str, path_to_pth_weights: str, device: str
) -> Any:
    """Метод для загрузки класификатора

    Args:
        name_of_classifier (str): _description_
        path_to_pth_weights (str): _description_
        device (str): _description_

    Returns:
        Any: _description_
    """
    pass


def arguments_parser() -> argparse.Namespace:
    """Парсер аргументов

    Returns:
        argparse.Namespace: _description_
    """
    parser = argparse.ArgumentParser(
        description="Скрипт для выполнения классификатора на единичном изображении или папке с изображениями"
    )
    parser.add_argument(
        "--name_of_classifier", "-nc", type=str, help="Название классификатора"
    )
    parser.add_argument(
        "--path_to_weights",
        "-wp",
        type=str,
        help="Путь к PTH-файлу с весами классификатора",
    )
    parser.add_argument(
        "--path_to_content",
        "-cp",
        type=str,
        help="Путь к одиночному изображению/папке с изображениями",
    )
    parser.add_argument(
        "--use_cuda",
        "-uc",
        action="store_true",
        help="Использовать ли CUDA для инференса",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Основная логика работы с классификатором"""
    args = arguments_parser()

    name_of_classifier = args.name_of_classifier
    path_to_weights = args.path_to_weights
    path_to_content = args.path_to_content
    use_cuda = args.use_cuda

    print(f"Name of classifier: {name_of_classifier}")
    print(f"Path to content: {path_to_content}")
    print(f"Path to weights: {path_to_weights}")

    if use_cuda:
        print("Device: CUDA")
    else:
        print("Device: CPU")


if __name__ == "__main__":
    main()
