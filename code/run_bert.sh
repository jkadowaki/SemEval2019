export BERT_BASE_DIR=bert/cased_L-12_H-768_A-12
export DATA_DIRECTORY=data/fold0
export CUDA_VISIBLE_DEVICES=0

python bert/run_classifier_modified.py             \
--task_name=cola                                   \
--do_train=true                                    \
--do_eval=true                                     \
--do_predict=true                                  \
--data_dir=$DATA_DIRECTORY                         \
--vocab_file=$BERT_BASE_DIR/vocab.txt              \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   \
--max_seq_length=128                               \
--train_batch_size=32                              \
--learning_rate=2e-5                               \
--num_train_epochs=50                              \
--save_checkpoints_steps=373                       \ # 11916 examples / 32 batch_size
--output_dir=$DATA_DIRECTORY/results
