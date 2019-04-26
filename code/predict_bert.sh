export BERT_BASE_DIR=bert/cased_L-12_H-768_A-12
export DATA_DIR=data/fold9
export RESULTS_DIR=$DATA_DIR/results

export CUDA_VISIBLE_DEVICES=0


for ckpt in $RESULTS_DIR/model.ckpt*.index
do
    echo $ckpt
    export CKPT_NAME=${ckpt%.index}
    export NUM=`cut -d'-' -f2 <<< $CKPT_NAME`
    export MODEL_DIR=$RESULTS_DIR/ckpt$NUM

    mkdir -p $MODEL_DIR
    mv $CKPT_NAME* $MODEL_DIR
    cp $RESULTS_DIR/graph.pbtxt $MODEL_DIR

    echo "model_checkpoint_path: \"`basename $CKPT_NAME`\" > $MODEL_DIR/checkpoint

    python bert/run_classifier_modified.py             \
    --task_name=cola                                   \
    --do_train=false                                   \
    --do_eval=true                                     \
    --do_predict=true                                  \
    --data_dir=$DATA_DIR                               \
    --vocab_file=$BERT_BASE_DIR/vocab.txt              \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$MODEL_DIR/`basename $CKPT_NAME` \
    --max_seq_length=128                               \
    --output_dir=$MODEL_DIR

    mv $MODEL_DIR/eval_results.txt  $RESULTS_DIR/eval_results_$NUM.txt
    mv $MODEL_DIR/eval.tf_record    $RESULTS_DIR/eval_$NUM.tf_record
    mv $MODEL_DIR/predict.tf_record $RESULTS_DIR/predict_$NUM.tf_record
    mv $MODEL_DIR/test_results.tsv  $RESULTS_DIR/test_results_$NUM.tsv

    tar -zcvf $DATA_DIR/epoch_results.tar.gz                            \
        $RESULTS_DIR/test_results_*.tsv $RESULTS_DIR/eval_results_*.txt

done

