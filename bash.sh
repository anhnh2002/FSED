for j in 5 10
do
    for m in 10
    do
        for n in 1
        do
            CUDA_LAUNCH_BLOCKING=1 python train.py \
                --data-root ./data_incremental \
                --dataset ACE \
                --backbone bert-base-uncased \
                --lr 2e-5 \
                --decay 1e-4 \
                --no-freeze-bert \
                --shot-num $j \
                --batch-size 1 \
                --device cuda:0 \
                --device2 cuda:0 \
                --log \
                --log-dir ./outputs/log_incremental/temp7_submax/first_wo_UCL+TCL/ \
                --log-name ashuffle_lnone_r5 \
                --dweight_loss \
                --rep-aug mean \
                --distill mul \
                --epoch 20 \
                --class-num $m \
                --single-label \
                --cl-aug shuffle \
                --aug-repeat-times 5 \
                --joint-da-loss none \
                --sub-max \
                --cl_temp 0.07 \
                --tlcl \
                --ucl \
                --skip-first-cl ucl+tlcl \
                --perm-id $n
        done
    done
done


