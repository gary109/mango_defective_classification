import os

def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)
        
CMD='./predict_pitch.py' 
MODEL='pix2pix'
CLASS='singing_transcription'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=256 # scale images to this size
CROP_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image
NO_FLIP='--no_flip'
DATAROOT='./datasets/original/AIcup_testset_ok/'
# DATAROOT='../../BicycleGAN/datasets/datasets/Singing_transcription'
CHECKPOINTS_DIR='./pretrained_models/%s/'%(CLASS)
PHASE='--phase predict'
CPU='--gpu_ids -1'
GPU_ID=0   # gpu id
NUM_TEST=100000 # number of input images duirng test
# number of samples per input images
NUM_SAMPLES='--n_samples 1' 
# USE_DROPOUT='--use_dropout'
NAME='%s_%s'%(CLASS,MODEL)
# RESULTS_DIR='./results/%s/'%(NAME)
RESULTS_DIR='./results/'
NET_G='--netG unet_256'
DATASET_MODE='--dataset_mode single'#'aligned'

os.environ['CMD']=str(CMD)
os.environ['MODEL']=str(MODEL)
os.environ['CLASS']=str(CLASS)
os.environ['DIRECTION']=str(DIRECTION)
os.environ['LOAD_SIZE']=str(LOAD_SIZE)
os.environ['CROP_SIZE']=str(CROP_SIZE)
os.environ['INPUT_NC']=str(INPUT_NC)
os.environ['NO_FLIP']=str(NO_FLIP)
os.environ['DATAROOT']=str(DATAROOT)
os.environ['CHECKPOINTS_DIR']=str(CHECKPOINTS_DIR)
os.environ['PHASE']=str(PHASE)
os.environ['CPU']=str(CPU)
os.environ['GPU_ID']=str(GPU_ID)
os.environ['NUM_TEST']=str(NUM_TEST)
os.environ['NUM_SAMPLES']=str(NUM_SAMPLES)
os.environ['NAME']=str(NAME)
os.environ['RESULTS_DIR']=str(RESULTS_DIR)
os.environ['NET_G']=str(NET_G)
os.environ['DATASET_MODE']=str(DATASET_MODE)

run('cp -rf ../../BicycleGAN/checkpoints/singing_transcription/singing_transcription_pix2pix/latest_net_* ./pretrained_models/singing_transcription/singing_transcription_pix2pix/')

run(' CUDA_VISIBLE_DEVICES=$GPU_ID python $CMD \
  --dataroot $DATAROOT \
  --results_dir $RESULTS_DIR \
  --checkpoints_dir $CHECKPOINTS_DIR \
  --name $NAME \
  --model $MODEL \
  --serial_batches \
  $PHASE \
  $DATASET_MODE  \
  --direction $DIRECTION \
  --load_size $LOAD_SIZE \
  --crop_size $CROP_SIZE \
  --input_nc $INPUT_NC \
  --num_test $NUM_TEST \
  $NUM_SAMPLES \
  --center_crop \
  --no_encode \
  $NO_FLIP \
  $USE_DROPOUT \
  $NET_G \
  $CPU')

print ('Done')