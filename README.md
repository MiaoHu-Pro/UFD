
This project is to conduct Knowledege Graph Embedding using entity/relation description.

FB15K237 
python run_bert_triple_classifier.py --task_name kg  --do_train  
--do_eval --do_predict --data_dir ./data/FB15K237  --bert_model bert-base-cased --max_seq_length 512 
--train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./output_FB15K237/  
--gradient_accumulation_steps 1  --eval_batch_size 512


python train_Bert.py --dataset FB15K237 --hidden_size 100 --num_of_filters 128 --neg_num 1 --valid_step 50 --nbatches 100 --num_epochs 300 --learning_rate 0.01 --lmbda 0.1 --model_name FB15K237_lda-0.1_nneg-10_nfilters-128_lr-0.01 --mode train --bert_model bert-base-cased --max_seq_length 400


3 Agu :
finish the scoring function

4 Agu :
to modify  Loss function - using MarginLoss






python3 main.py --task_name kg --do_train --do_eval  --do_predict --data_dir ./data/FB15k237  --bert_model bert-base-cased  --max_seq_length 150 --train_batch_size 16  --learning_rate 5e-5  --num_train_epochs 5.0  --output_dir ./output_FB15k-237/  --gradient_accumulation_steps 1  --eval_batch_size 1500


python3 main.py --task_name kg --do_train --do_eval --do_predict --data_dir ./data/WN18RR --bert_model bert-base-cased --max_seq_length 300 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 5.0 --output_dir ./output_WN18RR/  --gradient_accumulation_steps 1  --eval_batch_size 32
