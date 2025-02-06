# Training on 1 v100
sh /OntoGene/run_main.sh \
      --model /OntoGene/data/output_data/model/promotercore/OntoGeneModel \
      --output_file /OntoGene/data/datasets/promotercore \
      --task_name promotercore \
      --do_train False \
      --epoch 8 \
      --mean_output True \
      --optimizer  SGD \
      --per_device_batch_size 8 \
      --gradient_accumulation_steps 16 \
      --eval_step 100 \
      --eval_batchsize 1 \
      --warmup_ratio 0.01 \
      --frozen_bert False  \

