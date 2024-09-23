import os

# OpenAI Key
OPENAI_KEY = ""

# Checkpoints & Dataset root
ROOT=""
DATASET_ROOT=os.path.join(ROOT, "dataset")
CKPT_ROOT=os.path.join(ROOT, "checkpoints")
SAVE_ROOT=os.path.join(ROOT, "Phantom")
MODEL_7B="BK-Lee/Phantom-7B"
MODEL_3_8B="BK-Lee/Phantom-3.8B"
MODEL_1_8B="BK-Lee/Phantom-1.8B"
MODEL_0_5B="BK-Lee/Phantom-0.5B"

# Json files for Evaluation
SQA = "ScienceQA/problems.json"
SQA_SPLIT = "ScienceQA/pid_splits.json"
POPE_POPULAR = "POPE/coco_pope_popular.json"
POPE_ADVERSARIAL = "POPE/coco_pope_adversarial.json"
POPE_RANDOM = "POPE/coco_pope_random.json"
MME = "MME_Benchmark_release_version/llava_mme.json"
MME_DIR = "MME_Benchmark_release_version"
MMBENCH = "MMBench/MMBench_TEST_EN_legacy.tsv"
MMBENCH_CN = "MMBench/MMBench_TEST_CN_legacy.tsv"
MMBENCH_DEV = "MMBench/mmbench_dev_20230712.tsv"
MMBENCH_CN_DEV = "MMBench/mmbench_dev_cn_20231003.tsv"
MMVET = "mm-vet/mm-vet.json"
MMVET_V2 = "mm-vet-v2/mm-vet-v2.json"
MATHVISTA = "MathVista/testmini-00000-of-00001-725687bf7a18d64b.parquet"
AI2D = "ai2d/ai2d_test.json"
HALLUSIONBENCH = "HallusionBench/HallusionBench.json"
CHARTQA = "chartqa/test/test_augmented.json"
SEED = "SEED-Bench/SEED-Bench.json"
SEED_2_PLUS = "SEED-Bench-2-plus/SEED-Bench-2-plus-text-rich.json"
LLAVA = "llava-bench-in-the-wild/questions.jsonl"
LLAVA_WILDER = "LLaVA-Bench-Wilder/small/test-00000-of-00001.parquet"
MMSTAR = "MMStar/mmstar.parquet"
MATHVERSE = "MathVerse/testmini.json"
MATHVERSE_TEXT_ONLY = "MathVerse/testmini_text_only.json"
VISUALWEBBENCH = "VisualWebBench/*"
BLINK = "BLINK/*/val-*"
CVBENCH = "CV-Bench/test.parquet"

# Available evaluation datasets
EVAL_DATASETS = ["sqa", "ai2d", "chartqa", "seed", "seed-2-plus", "pope", "hallusionbench", "mme", \
                 "mathvista", "mmbench_dev", "mmbench_cn_dev", "mmvet", "mmvet-v2", "llava", "llava_wilder", "mmstar", "mathverse", "blink", "cvbench", "visualwebbench"]