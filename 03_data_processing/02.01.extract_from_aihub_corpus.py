import os
import glob
import json
import zipfile
import re
from multiprocessing import Pool, current_process
import shutil
from tqdm import tqdm  # tqdm 라이브러리를 임포트


def extract_zip_file(zip_file_info):
    zip_file, extract_path = zip_file_info
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def extract_all_zip_files(base_path):
    zip_files = glob.glob(f"{base_path}/**/*.zip", recursive=True)
    zip_file_info_list = [
        (zip_file, os.path.dirname(zip_file)) for zip_file in zip_files
    ]

    # ZIP 파일을 병렬로 압축 해제
    with Pool(processes=os.cpu_count()) as pool:
        list(
            tqdm(
                pool.imap(extract_zip_file, zip_file_info_list),
                total=len(zip_file_info_list),
                desc="Extracting ZIP files",
            )
        )


def find_all_json_files(base_path):
    return glob.glob(f"{base_path}/**/*.json", recursive=True)


def contains_korean(text):
    return bool(re.search("[가-힣]", text))


def find_korean_values(data, result):
    if isinstance(data, dict):
        for value in data.values():
            find_korean_values(value, result)
    elif isinstance(data, list):
        for item in data:
            find_korean_values(item, result)
    elif isinstance(data, str) and contains_korean(data):
        result.append({"text": data})


def safe_load_json_file(json_file):
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file}: {e}")
        return None


def process_json_file(args):
    json_file, temp_dir = args
    data = safe_load_json_file(json_file)
    temp_file_path = None  # 초기화 수정

    if data:
        result = []
        find_korean_values(data, result)
        if result:
            temp_file_path = os.path.join(
                temp_dir, f"temp_{current_process().pid}.jsonl"
            )
            with open(temp_file_path, "a", encoding="utf-8") as file:
                for item in result:
                    json_record = json.dumps(item, ensure_ascii=False)
                    file.write(json_record + "\n")

    # temp_file_path가 성공적으로 생성되었는지 확인
    if temp_file_path:
        return temp_file_path
    else:
        return None


def merge_temp_files(temp_files, final_path):
    with open(final_path, "w", encoding="utf-8") as final_file:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as f:
                    shutil.copyfileobj(f, final_file)
                os.remove(temp_file)  # Remove temp file after merging


def main(base_path, target_path, top_dir):
    extract_all_zip_files(base_path)
    json_files = find_all_json_files(base_path)
    temp_dir = os.path.join(target_path, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    args_list = [(json_file, temp_dir) for json_file in json_files]

    with Pool(processes=os.cpu_count()) as pool:
        temp_files = list(
            tqdm(
                pool.imap(process_json_file, args_list),
                total=len(args_list),
                desc="Processing JSON files",
            )
        )

    final_file_name = f"{top_dir}.jsonl"
    final_path = os.path.join(target_path, final_file_name)
    merge_temp_files(set(filter(None, temp_files)), final_path)


if __name__ == "__main__":
    base_path = "/mnt/t7/dnn_data/korean_data/data"
    target_path = "/mnt/t7/dnn/llm_practicing/korean_data"

    top_dir_list = [
        dir
        for dir in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, dir))
    ]

    for top_dir in tqdm(top_dir_list):
        base_target_path = os.path.join(base_path, top_dir)
        main(base_target_path, target_path, top_dir)
        print(top_dir, "is done!!!")
