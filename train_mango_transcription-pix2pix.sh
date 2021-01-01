set -ex
MODEL='pix2pix'

# dataset details
CLASS='mango_transcription'
NO_FLIP='--no_flip'
DIRECTION='AtoB'
LOAD_SIZE=256
CROP_SIZE=256
INPUT_NC=3
#-------------------------------------------------------
NITER=21
NITER_DECAY=379
#-------------------------------------------------------
# EPOCH_COUNT='--epoch_count 2365'
CONTINUE_TRAIN='--continue_train'
#-------------------------------------------------------
DISPLAY_PORT=8097
#-------------------------------------------------------
GAN_MODE='lsgan'
USE_DROPOUT='--use_dropout'
# LAMBDA_L1='--lambda_L1 100'
# LAMBDA_L1='--lambda_L1 50000'
# LAMBDA_L1='--lambda_L1 90000'
# LAMBDA_L1='--lambda_L1 100000'
# LAMBDA_L1='--lambda_L1 120000'
# LAMBDA_L1='--lambda_L1 200000'
LAMBDA_L1='--lambda_L1 250000'
LAMBDA_GAN='--lambda_GAN 1'
NORM='--norm batch'
# NORM='--norm instance'

# LR_POLICY='--lr_policy linear'
LR_POLICY='--lr_policy cosine'
NEF='--nef 64'
NGF='--ngf 64'
NDF='--ndf 64'
NET_D='--netD basic_256_multi'
# NET_D='--netD unet_256'
NET_G='--netG unet_256'
DATAROOT='./datasets/outputs'
NUM_DS='--num_Ds 2'
# USE_SAME_D='--use_same_D'
USE_SAME_D=''

NL='--nl relu'
# NL='--nl lrelu'
# NL='--nl elu'
#-------------------------------------------------------
SAVE_EPOCH_FREQ='--save_epoch_freq 5'
# SAVE_EPOCH_FREQ= '--save_epoch_freq 20'
SAVE_LATEST_FREQ='--save_latest_freq 2500'
#-------------------------------------------------------
CHECKPOINTS_DIR=./checkpoints/${CLASS}/
NAME=${CLASS}_${MODEL}
# ###################
# # Single GPU 0    #
# ###################
# CPU='--gpu_ids 0'
# GPU_ID=0
# DISPLAY_ID=10
# NUM_THREADS=0
# BATCH_SIZE=96

# ###################
# # Multi GPU 0,1  #
# ###################
CPU='--gpu_ids 0,1'
GPU_ID=0,1
DISPLAY_ID=10
NUM_THREADS=1
# BATCH_SIZE=192
BATCH_SIZE=8
# NUM_THREADS=0
# BATCH_SIZE=184


# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --display_port ${DISPLAY_PORT} \
  --batch_size ${BATCH_SIZE} \
  --gan_mode ${GAN_MODE} \
  --num_threads ${NUM_THREADS} \
  ${NL} \
  ${USE_SAME_D} \
  ${NUM_DS} \
  ${LR_POLICY} \
  ${SAVE_EPOCH_FREQ} \
  ${SAVE_LATEST_FREQ} \
  ${USE_DROPOUT} \
  ${NEF} \
  ${NGF} \
  ${NDF} \
  ${NORM} \
  ${CONTINUE_TRAIN} \
  ${LAMBDA_L1} \
  ${LAMBDA_GAN} \
  ${NET_G} \
  ${NET_D} \
  ${EPOCH_COUNT} \
  ${CONTINUE_TRAIN} \
  ${CPU} \
  ${NO_FLIP} \
  
  
