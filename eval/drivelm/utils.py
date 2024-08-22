import re
import torch
import json
import random
from eval.drivelm.evaluation import evaluation_suit


def format_sentence_for_drivelm(sentence):
    # Function to capitalize the first letter of a sentence and after any '.', '!', or '?'
    def capitalize_after_punctuation(text):
        try:
            return re.sub(
                r"(?<=[.!?])\s*(\w)",
                lambda m: m.group(1).upper(),
                text[0].upper() + text[1:],
            )
        except:
            return text

    # Remove unnecessary whitespaces between words and punctuation marks
    def remove_unnecessary_spaces(text):
        try:
            text = re.sub(r"\s*([.,!?])\s*", r"\1 ", text)
            text = re.sub(r"\s+", " ", text).strip()
        finally:
            return text

    # Format the special block <,,,>
    def format_c_tag(text):
        def format_block(match):
            parts = match.group(0).strip("<>").split(",")
            formatted_parts = (
                [parts[0].strip()]
                + [parts[1].strip().replace(" ", "").upper()]
                + [x.strip() for x in parts[2:]]
            )
            return f"<{','.join(formatted_parts)}>"

        try:
            return re.sub(r"<[^>]*>", format_block, text)
        except:
            return text

    def format_multiple_choice(text):
        try:
            if text.startswith(("A", "B", "C", "D")) and len(text) < 10:
                return text[0]
            return text
        except:
            return text

    def format_yes_no(text):
        try:
            if len(text) < 10:
                if text.startswith("Yes"):
                    return text[:3] + "."
                elif text.startswith("No ") or text == "No":
                    return text[:2] + "."
            return text
        except:
            return text

    def format_id(text):
        try:
            text = re.sub(r"\bids\b", "IDs", text)
            text = re.sub(r"\bid\b", "ID", text)
            return text
        except:
            return text

    # Apply transformations
    sentence = remove_unnecessary_spaces(sentence)
    sentence = capitalize_after_punctuation(sentence)
    sentence = format_c_tag(sentence)
    sentence = format_multiple_choice(sentence)
    sentence = format_yes_no(sentence)
    sentence = format_id(sentence)
    return sentence


def generate_drivelm_output(model, data_loader, epoch, device):
    print("Generating output in format for DriveLM...")

    model.eval()
    with torch.no_grad():
        with open(f"./eval/drivelm/outputs/output_{epoch}.json", "w") as out_json:
            id_list = []
            out_dicts = []
            for i, (
                bev,
                question,
                answer,
                sample_token,
                scene_token,
                det,
                obj,
            ) in enumerate(data_loader):
                print(f"\r{i}/{len(data_loader)}", end="")

                bev = bev.to(device, non_blocking=True)
                output = model.generate(question, bev=bev, det=det, obj=obj)
                output = format_sentence_for_drivelm(output[0])

                out_dict = {}
                increment_add_to_dict(
                    id_list,
                    out_dict,
                    f"{scene_token[0]}_{sample_token[0]}",
                    question[0],
                    answer[0],
                    output,
                    0,
                )
                out_dicts.append(out_dict)

            json.dump(out_dicts, out_json, indent=4)

    print("\nOutput generated!")


def increment_add_to_dict(ids, d, id, q, gt_a, a, i):
    if f"{id}_{i}" in ids:
        increment_add_to_dict(ids, d, id, q, gt_a, a, i + 1)
    else:
        ids.append(f"{id}_{i}")
        d["id"] = f"{id}_{i}"
        d["question"] = q
        d["gt_answer"] = gt_a
        d["answer"] = a


def calculate_final_score(eval_out):
    # Normalize to 0-1 and combine the scores: chatgpt, language, match, accuracy
    scores = []
    weights = [0.4, 0.2, 0.2, 0.2]

    # chatGPT
    score = eval_out["chatgpt"] / 100.0
    scores.append(score)

    # language
    score = 0
    for idx, key in enumerate(eval_out["language"].keys()):
        if idx < 4:
            score += eval_out["language"][key] / 4.0 / 3.0
        elif idx == 4:
            score += eval_out["language"][key] / 3.0
        else:
            score += eval_out["language"][key] / 10.0 / 3.0

    scores.append(score)

    # match
    score = eval_out["match"] / 100.0
    scores.append(score)

    # accuracy
    score = eval_out["accuracy"]
    scores.append(score)

    final_score = sum([x * y for x, y in zip(scores, weights)])
    return final_score


def drivelm_evaluation(pred_file_path, test_file_path):
    print("Running DriveLM evaluation...")

    with open(pred_file_path, "r") as f:
        pred_file = json.load(f)
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}

    with open(test_file_path, "r") as f:
        test_file = json.load(f)

    evaluation = evaluation_suit()
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]["key_frames"]

        for frame_id in scene_data.keys():
            frame_data_qa = scene_data[frame_id]["QA"]
            first_flag = True

            for i, qa in enumerate(
                frame_data_qa["perception"]
                + frame_data_qa["prediction"]
                + frame_data_qa["planning"]
                + frame_data_qa["behavior"]
            ):
                question = qa["Q"]
                GT = qa["A"]
                tag = qa["tag"]
                idx = scene_id + "_" + frame_id + "_" + str(i)
                predict = pred_file[idx]["answer"]
                # assert pred_file[idx]["gt_answer"] == GT, print(pred_file[idx]["gt_answer"], GT)
                if first_flag:
                    first_flag = False
                    evaluation.set_graph(predict, GT)
                    evaluation.forward(tag, predict, GT)
                else:
                    if evaluation.eval_graph(question):
                        res = evaluation.forward(tag, predict, GT)

    eval_results = evaluation.evaluation()
    final_sc = calculate_final_score(eval_results)
    eval_results["final_score"] = final_sc

    print("DriveLM evaluation complete!")
    print("Results:\n", eval_results)

    return eval_results


def split_dataset():
    train_file_path = "./data_thesis/QA_dataset_nus/v1_1_train_nus.json"
    train_converted_file_path = (
        "./data_thesis/QA_dataset_nus/v1_1_train_nus_converted.json"
    )

    train_path = "./data_thesis/QA_dataset_nus/train.json"
    test_path = "./data_thesis/QA_dataset_nus/test.json"
    val_path = "./data_thesis/QA_dataset_nus/val.json"

    train_conv_path = "./data_thesis/QA_dataset_nus/train_converted.json"
    test_conv_path = "./data_thesis/QA_dataset_nus/test_converted.json"
    val_conv_path = "./data_thesis/QA_dataset_nus/val_converted.json"

    with open(train_file_path, "r") as train_file:
        train_json = json.load(train_file)

    with open(train_converted_file_path, "r") as train_converted_file:
        train_converted_json = json.load(train_converted_file)

    scene_ids = list(train_json.keys())

    random.shuffle(scene_ids)

    # Calculate split indices
    num_scenes = len(scene_ids)
    split1 = int(0.75 * num_scenes)
    split2 = split1 + int(0.2 * num_scenes)

    # Split the list
    train_ids = scene_ids[:split1]
    test_ids = scene_ids[split1:split2]

    train = {}
    test = {}
    val = {}
    train_conv = {}
    test_conv = {}
    val_conv = {}
    i = 0

    for scene_id in train_json.keys():
        i += 1
        print(f"Scene{i}/{num_scenes}", end="")

        if scene_id in train_ids:
            train[scene_id] = train_json[scene_id]
            train_conv[scene_id] = train_converted_json[scene_id]
        elif scene_id in test_ids:
            test[scene_id] = train_json[scene_id]
            test_conv[scene_id] = train_converted_json[scene_id]
        else:
            val[scene_id] = train_json[scene_id]
            val_conv[scene_id] = train_converted_json[scene_id]

    with open(train_path, "w") as f:
        json.dump(train, f, indent=4)

    with open(train_conv_path, "w") as f:
        json.dump(train_conv, f, indent=4)

    with open(test_path, "w") as f:
        json.dump(test, f, indent=4)

    with open(test_conv_path, "w") as f:
        json.dump(test_conv, f, indent=4)

    with open(val_path, "w") as f:
        json.dump(val, f, indent=4)

    with open(val_conv_path, "w") as f:
        json.dump(val_conv, f, indent=4)
