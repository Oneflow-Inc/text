set -aux


DATA_PATH="datasets"
if [ ! -d "$DATA_PATH" ]; then
    wget http://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/FlowText/SimCSE/datasets.zip
    unzip datasets.tar.gz
fi

TASK='sup'
DEVICE='cuda'
EPOCHS=1
LR=3e-5
BATCH_SIZE=64
DROPOUT=0.3
POOLER_TYPE='last-avg'
MODEL_TYPE="bert-base-chinese"
PRETRAIND_PATH='/home/xiezipeng/text/pretrained_flow/bert-base-chinese-oneflow'
SAVE_PATH='./saved_model/simcse_sup'
TRAIN_DATA_PATH='./datasets/SNLI/train.txt'
DEV_DATA_PATH='./datasets/STS/cnsd-sts-dev.txt'
TEST_DATA_PATH='./datasets/STS/cnsd-sts-test.txt'


python3 main.py \
    --task $TASK \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --dropout $DROPOUT \
    --pooler_type $POOLER_TYPE \
    --model_type $MODEL_TYPE \
    --pretrained_path $PRETRAIND_PATH \
    --save_path $SAVE_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --dev_data_path $DEV_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
