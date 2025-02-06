cd ../

# Training on 1 v100
 sh /OntoGene/run_main.sh \
      --model /OntoGene/data/output_data/model/promotercore/OntoGeneModel \
      --output_file /OntoGene/data/datasets/strength \
      --task_name strength \
      --do_train False \
      --epoch 10 \
      --mean_output True \
      --optimizer  SGD \
      --per_device_batch_size 16 \
      --gradient_accumulation_steps 16 \
      --eval_step 100 \
      --eval_batchsize 1 \
      --warmup_ratio 0.08 \
      --frozen_bert False  \

