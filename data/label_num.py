import json

def count_pedestrians_jsonl(file_path):
    count = 0
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("label") == "Pedestrian":
                count += 1
    return count


if __name__ == "__main__":
    path = "/home/ximeng/Documents/SparseCLIP/data/kitti_triplet_train.jsonl"
    num = count_pedestrians_jsonl(path)
    print("Number of Pedestrian samples:", num)
