FILE=$1

function gdownload () { 
    if [[ -n ${2} ]]; then FNAME="-O ${2}" ; fi; wget -q --show-progress --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${1}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${1}" $FNAME && rm -rf /tmp/cookies.txt;
}

if [ $FILE == "celeba" ]; then

    # CelebA images
    URL=https://www.dropbox.com/s/ftcx1gf6tobtw08/celeba.zip?dl=0
    ZIP_FILE=./data/celeba.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE

elif [ $FILE == "pcam" ]; then
    # Download PatchCamelyon from Google Drive
    mkdir -p ./data/pcam/
    gdownload '1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_' ./data/pcam/test_x.h5.gz
    gdownload '1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_' ./data/pcam/test_y.h5.gz
    gdownload '1g04te-mWB_GvM4TFyhw3xdzrV8xTXPJO' ./data/pcam/train_mask.h5.gz
    gdownload '1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2' ./data/pcam/train_x.h5.gz
    gdownload '1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG' ./data/pcam/train_y.h5.gz
    gdownload '1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3' ./data/pcam/valid_x.h5.gz
    gdownload '1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO' ./data/pcam/valid_y.h5.gz
    gunzip ./data/pcam/*.gz

elif [ $FILE == "brats" ]; then

    # BRATS 2013 processed synthetic images
    URL=https://www.dropbox.com/s/057db0dqp4pymoa/brats.zip?dl=0
    ZIP_FILE=./data/brats.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE

elif [ $FILE == 'pretrained_celeba_128' ]; then

    # Fixed-Point GAN trained on CelebA (Black_Hair, Blond_Hair, Brown_Hair, Male, Young), 128x128 resolution
    URL=https://www.dropbox.com/s/es0d8q0qk29egci/pretrained_celeba_128.zip?dl=0
    ZIP_FILE=./pretrained_models/pretrained_celeba_128.zip
    mkdir -p ./pretrained_models/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./pretrained_models/
    rm $ZIP_FILE

elif [ $FILE == 'pretrained_brats_256' ]; then

    # Fixed-Point GAN trained on BRATS 2013 synthetic dataset, 256x256 resolution
    URL=https://www.dropbox.com/s/knfmeza0ikzo9ep/pretrained_brats_syn_256_lambda0.1.zip?dl=0
    ZIP_FILE=./pretrained_models/pretrained_brats_syn_256_lambda0.1.zip
    mkdir -p ./pretrained_models/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./pretrained_models/
    rm $ZIP_FILE

else
    echo "Available arguments are celeba, pcam, brats, pretrained_celeba_128, and pretrained_brats_256."
    exit 1
fi