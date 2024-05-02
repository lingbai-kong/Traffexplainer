echo "Run Explainer..."
for dataset in PEMSD8
do
    echo "Dataset: ${dataset}"
    for model in GTS
    do
        random=$RANDOM
        echo "Explaining ${model}_${random}..."
        python run_model.py --task=traffic_state_pred --model=Explainer --base_model=${model} --dataset=${dataset} --batch_size=128 --max_epoch=30 --learning_rate=0.001 --gpu_id=0 --saved_model=True > cmd_log/${dataset}-Explainer-${model}.log 2>&1
        # echo "Evaluating ${model}..."
        # python run_model.py --task=traffic_state_pred --model=Explainer --base_model=${model} --dataset=${dataset} --batch_size=64 --max_epoch=30 --learning_rate=0.001 --gpu_id=0 --saved_model=False --train=False --exp_id=${random} > cmd_log/${dataset}-Evaluate-${model}.log 2>&1
    done
done
echo "Finish"