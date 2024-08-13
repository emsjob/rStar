import sys

sys.path.append(".")
from common.helpers import read_json, save_json
from eval_src.Evaluator import *

import os, multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from argparse import ArgumentParser
from collections import defaultdict

def extract_trace(data_item, num_votes):
    res = []
    subanswer, tot, direct_answer = 0, 0, 0
    subanswer_directanswer, tot_directanswer, subanswer_tot, subanswer_tot_directanswer = 0, 0, 0, 0
    for item in data_item:
        i = 0
        trace = item["trace"] if "trace" in item else item
        rollout_id = item["rollout_id"] if "rollout_id" in item else 0
        if num_votes != -1 and rollout_id >= num_votes:
            continue
        while str(i) in trace:
            i += 1
        if "direct_answer" in trace[str(i-1)]:
            # res.append(trace[str(i-1)]["direct_answer"]["text"])
            if trace[str(i-1)]["tot_step"] != {}:
                if i != 1:
                    subanswer_tot_directanswer += 1
                    res.append(trace[str(i-1)]["direct_answer"]["text"])
                else:
                    tot_directanswer += 1
                    res.append(trace[str(i-1)]["direct_answer"]["text"])
            else:
                if i != 1:
                    subanswer_directanswer += 1
                    res.append(trace[str(i-1)]["direct_answer"]["text"])
                else:
                    direct_answer += 1
                    res.append(trace[str(i-1)]["direct_answer"]["text"])
        elif trace[str(i-1)]["tot_step"] != {}:
            j = 1
            while str(j) in trace[str(i-1)]["tot_step"]:
                j += 1

            if i == 1: # only tot
                tot += 1
                res.append(trace[str(i-1)]["tot_step"][str(j-1)])
            else:
                subanswer_tot += 1
                res.append(trace[str(i-1)]["tot_step"][str(j-1)])
        elif "subanswer" in trace[str(i-1)]:
            res.append(trace[str(i-1)]["subanswer"]["text"])
            subanswer += 1
        else:
            import pdb; pdb.set_trace()
    detail_count = [subanswer, tot, direct_answer, subanswer_directanswer, tot_directanswer, subanswer_tot, subanswer_tot_directanswer]
    assert num_votes != -1 or sum(detail_count) == len(data_item)
    return res, detail_count

def extrace_completions(data_item):
    res = []
    for item in data_item:
        res.append(data_item[item]["model_solution"])
    return res

def eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, evaluator: Evaluator, vote_pool_dir=None, num_votes=-1):
    data_item = {}
    gold_answer = read_json(os.path.join(answer_sheets_dir, f"{example_id}.json"))["gold_answer"]
    data_1 = read_json(os.path.join(answer_sheets_dir, example_id.replace(" - Answer", " - Final Solutions") + ".json"))
    data_2 = read_json(os.path.join(answer_sheets_dir, example_id.replace(" - Answer", " - Rollout Solutions") + ".json"))

    data_1, count_1 = extract_trace(data_1, num_votes) # data_1 + data_2
    data_2, count_2 = extract_trace(data_2, num_votes) # data_1 + data_2
    data_1_2 = data_1 + data_2
    count_1_2 = [item_1 + item_2 for (item_1, item_2) in zip (count_1, count_2)]
    model_answer_1_2, _, _, _ = evaluator.find_most_confident_answer(data_1_2)
    result_1_2 = evaluator.check_answers_equiv(model_answer_1_2, gold_answer)
    data_item["correctness_1_2"] = result_1_2
    data_item["count_1_2"] = count_1_2

    return data_item


def eval_exp(exp_dir: str, dataset_name: str, num_votes: int = -1):
    answer_sheets_dir = os.path.join(exp_dir, "answer_sheets")
    vote_pool_dir = os.path.join(exp_dir, "vote_pool")

    example_ids = [f.replace(".json", "") for f in os.listdir(answer_sheets_dir) \
                   if f.endswith(" - Answer.json")]
    evaluator = eval(f"{dataset_name}Evaluator()")

    data_list = []
    for example_id in tqdm(example_ids):
        try:
            dta = eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, evaluator, vote_pool_dir, num_votes)
            data_list.append(dta)
        except:
            print(f"Error in {example_id}")

    # Calculate accuracy
    print("Solution Node + Terminal Node")
    accuracy = sum([item["accuracy"] for item in data_list]) / len(data_list)
    count = [round(sum([item["count"][i] for item in data_list]) / len(data_list), 2) for i in range(len(data_list[0]["count_1"]))]
    print(f"accuracy: {accuracy}, count: {count}")

    # Save eval results
    eval_result_dir = answer_sheets_dir.replace("run", "eval").replace("answer_sheets", "")
    os.makedirs(eval_result_dir, exist_ok=True)
    save_json(data_list, os.path.join(eval_result_dir, "eval_results.json"))
    analysis = {"accuracy": accuracy}
    save_json(analysis, os.path.join(eval_result_dir, "analysis.json"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--exp_dir_path", type=str, required=True)
    parser.add_argument("--num_votes", type=int, default=-1)
    args = parser.parse_args()

    eval_exp(args.exp_dir_path, args.dataset_name, num_votes=args.num_votes)