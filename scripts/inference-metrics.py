from pathlib import Path

from iohub import open_ome_zarr
from dl_mbl_gut import evaluation
import numpy as np
import napari
import pandas as pd
import torch
import matplotlib.pyplot as plt

visualize = True
if visualize is True:
    v = napari.Viewer()

run_name = "lr1e-6"

input_path = Path("/mnt/efs/dlmbl/G-bs/data/all-downsample-8x.zarr")
inference_path = Path(
    "/mnt/efs/dlmbl/G-bs/data/all-downsample-8x-predictions-" + run_name + ".zarr"
)
split_path = Path("/mnt/efs/dlmbl/G-bs/data/all-downsample-2x-split.csv")
output_path = Path("/mnt/efs/dlmbl/G-bs/data/results/guts/09-01/")

if not output_path.exists():
    output_path.mkdir(parents=True)

# Loading data
input_dataset = open_ome_zarr(input_path, mode="r", layout="hcs")
infer_dataset = open_ome_zarr(inference_path, mode="r", layout="hcs")
data_channel_name = "BF_fluor_path"
mask_channel_name = "label"
inference_channel_name = "prediction"

split_data = pd.read_csv(split_path)
print(split_data)

split_data = split_data[split_data["position"] == "20230728/infected/3"]

inference_thresholds = np.arange(0.25, 0.8, 0.05)
for inference_threshold in inference_thresholds:
    print(f"Inference threshold: {inference_threshold:.2f}")
    for _, row in split_data.iterrows():
        id = row["position"]
        best_slice = row["best-slice"]
        pos = input_dataset[id]

        data = pos[0][0, input_dataset.get_channel_index(data_channel_name)]
        mask = pos[0][0, input_dataset.get_channel_index(mask_channel_name)]
        infer = infer_dataset[id][0][
            0, infer_dataset.get_channel_index(inference_channel_name)
        ]

        # Compute metric
        mask = mask[:, : infer.shape[-2], : infer.shape[-1]]

        infer = torch.tensor(infer[None, None]).to("cuda")
        mask = torch.tensor(mask[None, None]).to("cuda")
        metric = 1 - evaluation.DiceCoefficient()(infer > inference_threshold, mask > 0)
        metric = metric.to("cpu").numpy()

        # print(f"{id}, dice : {metric:.2f}, label: {row['label']}")
        split_data.loc[split_data["position"] == id, "dice_coefficient"] = metric

        # Visualize
        if visualize is True:
            v.add_image(data, name="data")
            v.add_image(
                mask.to("cpu").numpy(),
                name="mask",
                opacity=0.15,
                colormap="green",
                blending="additive",
            )
            v.add_image(
                infer.to("cpu").numpy(),
                name="inference",
                opacity=0.5,
                colormap="red",
                blending="additive",
            )
            v.dims.current_step = (best_slice, 0, 0)

            input("Press Enter to continue...")
            v.layers.clear()

    split_data.to_csv(output_path / "split_data.csv", index=False)

    # Load data from file
    split_data = pd.read_csv(output_path / "split_data.csv")

    # Generate box plot
    plt.figure(figsize=(3, 4))
    column_order = ["train", "val", "test"]
    dice_coefficients = []
    for position, column in enumerate(column_order):
        data = split_data[split_data["label"] == column]
        dice_coefficient = data["dice_coefficient"]
        dice_coefficients.append(dice_coefficient)
        plt.boxplot(
            dice_coefficient, positions=[position + 1], widths=0.6, showfliers=False
        )
        y = dice_coefficient
        x = np.random.normal(position + 1, 0.04, size=len(y))
        plt.plot(x, y, "ro", alpha=0.6)

        mean_dice_coefficient = np.mean(dice_coefficient)
        print(f"Mean dice coefficient for {column}: {mean_dice_coefficient:.2f}")

    plt.xticks([1, 2, 3], ["train", "validate", "test"])
    plt.xlabel("")
    plt.ylabel("Dice Coefficient")
    plt.ylim(0, 1)

    plt.savefig(
        output_path / ("box_plot" + run_name + str(inference_threshold) + ".pdf"),
        bbox_inches="tight",
    )
    plt.close()
