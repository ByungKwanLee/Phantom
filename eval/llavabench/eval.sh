python eval/llavabench/eval_gpt_review_bench.py \
    --question questions.jsonl \
    --context context.jsonl \
    --rule eval/llavabench/rule.json \
    --answer-list \
        answers_gpt4.jsonl \
       Phantom-7b_llava_results.jsonl \
    --output \
        reviews_phantom-7b_llava_results.jsonl

python eval/llavabench/summarize_gpt_review.py -f reviews_phantom-7b_llava_results.jsonl
