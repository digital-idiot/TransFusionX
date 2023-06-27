import cv2
from typing import Any
import rasterio as rio
from typing import Dict
from typing import Union
from pathlib import Path
from affine import Affine
from typing import Optional
from typing import Sequence
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BasePredictionWriter


# noinspection PyAbstractClass
class PredictionWriter(BasePredictionWriter):
    def __init__(
            self,
            dst_dir: Union[str, Path],
            colormap: Optional[Dict] = None
    ):
        self._dst_dir = Path(dst_dir)
        self._colormap = colormap
        super().__init__(write_interval="batch")

    def write_on_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            prediction: Any,
            batch_indices: Optional[Sequence[int]],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        for im, it, ic, pr, lb, lt, lc in zip(
            prediction["image"],
            prediction["image_transform_list"],
            prediction["image_crs_list"],
            prediction["prediction"],
            prediction["label"],
            prediction["label_transfoms_list"],
            prediction["label_crs_list"]
        ):
            im = im[:4, :, :].detach().clone().cpu().numpy()
            im = im - im.min(axis=0)
            max_vals = im.max(axis=0)
            max_vals[max_vals == 0] = 1
            im = im / max_vals
            im = (255 * im).astype("ubyte")
            _, encoded_img = cv2.imencode(
                ".jpg", im.transpose(1, 2, 0), (cv2.IMWRITE_JPEG_QUALITY, 95)
            )
            im = cv2.imdecode(
                encoded_img, cv2.IMREAD_UNCHANGED
            ).transpose(2, 0, 1)
            it = Affine(
                a=it.t_11, b=it.t_12, c=it.t_14,
                d=it.t_21, e=it.t_22, f=it.t_24
            )
            image_meta = {
                "driver": "PNG",
                "dtype": im.dtype,
                "crs": ic,
                "transform": it,
                "count": im.shape[0],
                "height": im.shape[1],
                "width": im.shape[2],
            }
            iname = f"{batch_idx:04d}_IMG.{image_meta['driver'].lower()}"
            with rio.open((self._dst_dir / iname), "w", **image_meta) as idst:
                idst.write(im)
            pr = pr.detach().clone().cpu().numpy().astype("ubyte")
            prd_meta = {
                "driver": "PNG",
                "dtype": pr.dtype,
                "crs": ic,
                "transform": it,
                "count": pr.shape[0],
                "height": pr.shape[1],
                "width": pr.shape[2],
            }
            pname = f"{batch_idx:04d}_PRD.{prd_meta['driver'].lower()}"
            with rio.open(
                (self._dst_dir / pname), "w", **prd_meta
            ) as pdst:
                pdst.write(pr)
                if self._colormap is not None:
                    for i, cmap in self._colormap.items():
                        pdst.write_colormap(i, cmap)
            if lb is not None:
                lb = lb.detach().clone().cpu().numpy().astype("ubyte")
                lt = Affine(
                    a=lt.t_11, b=lt.t_12, c=lt.t_14,
                    d=lt.t_21, e=lt.t_22, f=lt.t_24
                )
                lbl_meta = {
                    "driver": "PNG",
                    "dtype": lb.dtype,
                    "crs": lc,
                    "transform": lt,
                    "count": lb.shape[0],
                    "height": lb.shape[1],
                    "width": lb.shape[2],
                }
                lname = f"{batch_idx:04d}_LBL.{lbl_meta['driver'].lower()}"
                with rio.open(
                    (self._dst_dir / lname), "w", **lbl_meta
                ) as ldst:
                    ldst.write(lb)
                    if self._colormap is not None:
                        for i, cmap in self._colormap.items():
                            ldst.write_colormap(i, cmap)
