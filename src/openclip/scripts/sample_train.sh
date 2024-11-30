MODEL_NAME='ViT-B-32'
LR=0.0005
DATA_DIR=""
LOGS_DIR=""
BZ=1024
EPOCHS=10
PRECISION=bf16
DEVICE=cuda:1

python src/main.py \
    --model_name ${MODEL_NAME} \
    --lr ${LR} \
    --data_dir ${DATA_DIR} \
    --batch_size ${BZ} \
    --epochs ${EPOCHS} \
    --precision ${PRECISION} \
    --device ${DEVICE} \
    --log_dir ${LOGS_DIR} \
    --train