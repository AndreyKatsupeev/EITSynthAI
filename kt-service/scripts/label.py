import supervision as sv

ds = sv.DetectionDataset.from_yolo(
    images_directory_path=f"train_all/images",
    annotations_directory_path=f"train_all/labels",
    data_yaml_path=f"data.yaml"
)

train_ds, valid_ds = ds.split(split_ratio=0.7,
                             random_state=42, shuffle=True)


train_ds.as_yolo(
    images_directory_path='train/images',
    annotations_directory_path='train/labels')

valid_ds.as_yolo(
    images_directory_path='valid/images',
    annotations_directory_path='valid/labels')