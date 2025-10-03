# configs/config_landmark_petface.py
CFG = dict(
    IMG_ROOT="/home/sagemaker-user/LostPetTest/petface_data/images",

    # 你上一步生成的 splits（或用仓库自带 splits_petface）
    TRAIN_CSV="splits_petface_custom/train.csv",
    VAL_QUERY_CSV="splits_petface_custom/val_query.csv",
    VAL_GALLERY_CSV="splits_petface_custom/val_gallery.csv",
    TEST_QUERY_CSV="splits_petface_custom/test_query.csv",
    TEST_GALLERY_CSV="splits_petface_custom/test_gallery.csv",

    LANDMARKS_JSON=None,          # 有 landmark 就填 json 路径；没有就保持 None

    # 训练超参
    IMG_SIZE=256,                 # 或 (256,256)
    BATCH_SIZE=64,
    BACKBONE="dinov3_vits16",           # 你仓库里的名字
    LR=3.5e-4,
    EPOCHS=10,

    # 输出
    OUT_DIR="experiments/runs/landmark/seed42",
)
