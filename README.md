# OntoGene:Knowledge-Enhanced BERT for Promoter Identification


 In this package, we provide the following resources: the source code for OntoGene, the pretrained model weights, and the fine-tuning code. 
 The necessary Python packages are listed in the `requirements.txt` file. Additionally, you need to run `pip install -e` . in the main folder. We provide a pretrained script bash `script/run_pretrain.sh`, and the pretrained weights are located in the folder `OntoGene/data/output_data/model/promotercore/OntoGeneModel`.  

The dataset for fine-tuning the model is placed in the folder `OntoGene/data/datasets/promotercore/promotercore`.

#### Downstream models

Running shell files: `bash script/promotercore.sh`, and the contents of shell files are as follow:

```shell
bash /OntoGene/run_main.sh \
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
```



