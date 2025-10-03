CFG = dict(
  BASE_CONFIG="configs/config_petface.py",     # 继承你现有宠物配置
  DATASET=dict(
    NAME="PetFaceWithKpts",
    TRAIN_CSV="splits_petface_custom/train.csv",
    VAL_QUERY_CSV="splits_petface_custom/val_query.csv",
    VAL_GALLERY_CSV="splits_petface_custom/val_gallery.csv",
    TEST_QUERY_CSV="splits_petface_custom/test_query.csv",
    TEST_GALLERY_CSV="splits_petface_custom/test_gallery.csv",
    # IMG_ROOT="data/petface/images",
    # ANN_CSV="data/petface/splits/train.csv",
    KPT_JSON="data/petface/landmarks.json",
    USE_PCA_ROI=True,
    PCA_CACHE_DIR="cache/pca_roi",
  ),
  PROCESSOR=dict(
    ALIGN=True,
    REF_POINTS="default_3points",
    OUT_SIZE=(224,224),
    PCA_THR=0.6,
  ),
  MODEL=dict(
    BACKBONE="dinov3_s",                 # 或 swinv2
    REGION_AWARE=True,
    REGION_POOL_SIZE=3,
    HEAD="GLAHead",
    EMBED_DIM=512
  ),
  LOSS=dict(
    METRIC="arcface",                    # 维持与现有一致
    LOCAL_CONSIST=False,                 # 先关
  ),
  TRAIN=dict(
    BATCH_SIZE=64,
    EPOCHS=60,
    LR=1e-4,
    FP16=True
  ),
  TEST=dict(
    SPLIT_CSV="data/petface/splits/val.csv"
  ),
  OUTPUT="outputs/petface_landmark_dino",
  OUT_DIR="outputs/petface_landmark_dino"
)
