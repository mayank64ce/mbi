# python aim_HE_simulate_l2_clear.py \
#     --dataset "../data/breast_train.csv" \
#     --domain "../data/breast-domain.json" \
#     --seed 0

python aim_HE_simulate_l2.py \
    --dataset "../data/breast_train.csv" \
    --domain "../data/breast-domain.json" \
    --seed 1

python aim_HE_simulate_l2.py \
    --dataset "../data/breast_train.csv" \
    --domain "../data/breast-domain.json" \
    --seed 2