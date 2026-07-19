# Results

Test protocol: each user's last 5 positives; models trained on train+validation; 5656 evaluated users, 3533-item catalog. ALS config: factors=64, reg=0.1, alpha=1.0, iterations=15 (selected on the validation slice).

| Model | Recall@10 | NDCG@10 | MAP@10 | Coverage@10 | Mean pop. rank |
|---|---|---|---|---|---|
| popularity | 0.0452 | 0.0362 | 0.0161 | 3.3% | 11 |
| item-knn | 0.0611 | 0.0511 | 0.0242 | 9.0% | 34 |
| als | 0.0771 | 0.0625 | 0.0289 | 31.5% | 183 |

ALS vs item-knn, Recall@10 difference: +0.0159 [+0.0125, +0.0195] (1000 bootstrap resamples over users).
