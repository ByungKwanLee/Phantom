MODEL="Phantom-7b"

python eval/llavabench/eval_gpt_review_bench.py \
    --question questions.jsonl \
    --rule rule.json \
    --answer-list \
        answers_gpt4.jsonl \
        ${MODEL}_llava_wilder_results.jsonl \
    --output \
        reviews_${MODEL}_llava_wilder_results.jsonl \
    --version llava_wilder

python eval/llavabench/summarize_gpt_review.py -f reviews_${MODEL}_llava_wilder_results.jsonl
