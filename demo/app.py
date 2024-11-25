import yaml
from typing import List
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

import cv2
import torch
from torchvision.ops import nms
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

from pdf2image import convert_from_bytes

from ScanSSD.detect_flow import MathDetector
from HybridViT.recog_flow import MathRecognition
from utils.p2l_utils import get_rolling_crops, postprocess

import streamlit


class DetectCfg:
    def __init__(self):
        self.cuda = True if torch.cuda.is_available() else False
        self.kernel = (1, 5)
        self.padding = (0, 2)
        self.phase = "test"
        self.visual_threshold = 0.8
        self.verbose = False
        self.exp_name = "SSD"
        self.model_type = 512
        self.use_char_info = False
        self.limit = -1
        self.cfg = "hboxes512"
        self.batch_size = 32
        self.num_workers = 4
        self.neg_mining = True
        self.log_dir = "logs"
        self.stride = 0.1
        self.window = 1200


class App:
    title = "Math Expression Recognition Demo \n\n Note: For Math Detection, we reuse the model from this repo [ScanSSD: Scanning Single Shot Detector for Math in Document Images](https://github.com/MaliParag/ScanSSD).\n\nThis demo aim to present the effciency of our method [A Hybrid Vision Transformer Approach for Mathematical Expression Recognition](https://ieeexplore.ieee.org/document/10034626) in recognizing math expression in document images."

    def __init__(self):
        self._model_cache = {}
        self.detect_model = MathDetector(
            "saved_models/math_detect/AMATH512_e1GTDB.pth", DetectCfg()
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_resizer = ResNetV2(
            layers=[2, 3, 3],
            num_classes=max((672, 192)) // 32,
            global_pool="avg",
            in_chans=1,
            drop_rate=0.05,
            preact=True,
            stem_type="same",
            conv_layer=StdConv2dSame,
        ).to(device)
        self.image_resizer.load_state_dict(
            torch.load("saved_models/resizer/image_resizer.pth", map_location=device)
        )
        self.image_resizer.eval()

    def detect_preprocess(self, img_list):
        if isinstance(img_list, Image.Image):
            img_list = [img_list]

        new_images = []

        for temp_image in img_list:
            img_size = 1280
            # convert image to numpy array
            temp_image = np.array(temp_image)
            img = cv2.resize(
                temp_image,
                (img_size, int(img_size * temp_image.shape[0] / temp_image.shape[1])),
            )
            new_images.append(img)

        return new_images

    def _get_model(self, name):
        if name in self._model_cache:
            return self._model_cache[name]

        with open("recog_cfg.yaml", "r") as f:
            recog_cfg = yaml.safe_load(f)

        model_cfg = {}
        model_cfg.update(recog_cfg["common"])
        model_cfg.update(recog_cfg[name])
        recog_model = MathRecognition(
            model_cfg, self.image_resizer if model_cfg["resizer"] else None
        )
        self._model_cache[name] = recog_model

        return recog_model

    def _get_boxes(self, img, temp_bb):
        temp_bb[0] = max(0, temp_bb[0] - int(0.05 * (temp_bb[2] - temp_bb[0])))
        temp_bb[1] = max(0, temp_bb[1] - int(0.05 * (temp_bb[3] - temp_bb[1])))
        temp_bb[2] = min(
            img.shape[1], temp_bb[2] + int(0.05 * (temp_bb[2] - temp_bb[0]))
        )
        temp_bb[3] = min(
            img.shape[0], temp_bb[3] + int(0.05 * (temp_bb[3] - temp_bb[1]))
        )

        # convert to int
        temp_bb = [int(x) for x in temp_bb]

        return temp_bb

    @torch.inference_mode()
    def math_detection(self, page_lst: List[np.ndarray]):
        res = []

        batch_size = 32
        threshold = 0.9
        iou = 0.1

        for idx, temp_image in enumerate(page_lst):
            crops_list, padded_crops_list, crops_info_list = get_rolling_crops(
                temp_image, stride=[128, 128]
            )

            scores_list = []
            wb_list = []
            for i in range(0, len(padded_crops_list), batch_size):
                batch = padded_crops_list[i : i + batch_size]
                window_borders, scores = self.detect_model.DetectAny(batch, threshold)
                scores_list.extend(scores)
                wb_list.extend(window_borders)

            # change crops to original image coordinates
            bb_list, s_list = postprocess(wb_list, scores_list, crops_info_list)

            # convert to torch tensors
            bb_torch = torch.tensor(bb_list).float()
            scores_torch = torch.tensor(s_list)

            # perform non-maximum suppression
            # check if bb_torch is empty
            if bb_torch.shape[0] == 0:
                res.append(([], []))
                continue

            indices = nms(bb_torch, scores_torch, iou)

            bb_torch = bb_torch[indices]
            new_bb_list = bb_torch.int().tolist()

            for i in range(len(new_bb_list)):
                save_name = (
                    "Page " + str(idx) + "-Expr " + str(i)
                    if len(page_lst) > 1
                    else "Expr " + str(i)
                )
                temp_bb = self._get_boxes(temp_image, new_bb_list[i][:])
                crop_expr = temp_image[temp_bb[1] : temp_bb[3], temp_bb[0] : temp_bb[2]]
                crop_expr = Image.fromarray(crop_expr)
                res.append((save_name, crop_expr))

        return res

    def math_recognition(self, model_name, res: List):
        model = self._get_model(model_name)
        final_res = []
        for item in res:
            name, crop_expr = item
            if isinstance(crop_expr, list):
                continue
            latex_str = model(crop_expr, name=name)
            final_res.append((name, crop_expr, latex_str))

        return final_res

    def __call__(self, model_name, image_list, use_detect):
        # Detect
        if use_detect:
            new_images = self.detect_preprocess(image_list)
            res = self.math_detection(page_lst=new_images)
        else:
            res = [("latex_pred", image_list[0])]
        # Recog
        final_res = self.math_recognition(model_name, res)
        display_name, origin_img, latex_pred = tuple(
            [list(item) for item in zip(*final_res)]
        )
        return display_name, origin_img, latex_pred


def api():
    app = App()
    streamlit.set_page_config(page_title="Thesis Demo", layout="wide")
    streamlit.title(f"{app.title}")
    streamlit.markdown(
        f"""
            To use this interactive demo and reproduced models:
            1. Select what type of input data you want to get prediction.
            2. Upload your own image or pdf file (or select from the given examples).
            3. If input file is in pdf format, choose start page and end page.
            4. Click **Extract**.

            **Note: Current version of this demo only support single file upload for both Image and PDF option.**
        """
    )

    # model_name = streamlit.radio(
    #     label='The Math Recognition model to use',
    #     options=app.models
    # )

    extract_option = streamlit.radio(
        label="Select type of input for prediction",
        options=("Math expression image only", "Full document image"),
    )

    uploaded_file = streamlit.file_uploader(
        "Upload an image/pdf file",
        type=["png", "jpg", "pdf"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        if Path(uploaded_file.name).suffix == ".pdf":
            bytes_data = uploaded_file.read()

            image_lst = convert_from_bytes(bytes_data, dpi=160, grayscale=True)
            image_lst = [img.convert("RGB") for img in image_lst]

            container = streamlit.container()
            range_cols = container.columns(2)
            start_page = range_cols[0].number_input(
                label="Start page", min_value=0, max_value=len(image_lst) - 2
            )
            end_page = range_cols[1].number_input(
                label="End page", min_value=1, max_value=len(image_lst) - 1
            )

            if start_page <= end_page:
                image_lst = image_lst[start_page : end_page + 1]
                cols = streamlit.columns(len(image_lst))
                for i in range(len(cols)):
                    with cols[i]:
                        img_shape = image_lst[i].size
                        streamlit.image(
                            image_lst[i],
                            width=1024,
                            caption=f"Page: {str(i)} Image shape: {str(img_shape)}",
                            use_column_width="auto",
                        )
        else:
            image = Image.open(uploaded_file).convert("RGB")
            image_lst = [image]
            img_shape = image.size
            streamlit.image(image, width=1024, caption="Image shape: " + str(img_shape))
    else:
        streamlit.text("\n")

    if streamlit.button("Extract"):
        if uploaded_file is not None and image_lst is not None:
            with streamlit.spinner("Computing"):
                try:
                    use_detect = True
                    if extract_option == "Math expression image only":
                        use_detect = False
                        model_name = "version1"
                    else:
                        model_name = "version2"

                    display_name, origin_img, latex_code = app(
                        model_name, image_lst, use_detect
                    )

                    if Path(uploaded_file.name).suffix == ".pdf":
                        page_dict = defaultdict(list)
                        for name, img, pred in zip(
                            display_name, origin_img, latex_code
                        ):
                            name_components = name.split("-")
                            if len(name_components) <= 1:
                                page_name = "Page0"
                            else:
                                page_name = name_components[0]
                            page_dict[page_name].append((img, pred))

                        tab_lst = streamlit.tabs(list(page_dict.keys()))

                        for tab, page_name in zip(tab_lst, list(page_dict.keys())):
                            for idx, item in enumerate(page_dict[page_name]):
                                container = tab.container()
                                col_latex, col_render, col_org = container.columns(
                                    3, gap="large"
                                )

                                if idx == 0:
                                    col_latex.header("Predicted LaTeX")
                                    col_render.header("Rendered Image")
                                    col_org.header("Cropped Image")

                                render_latex = f"$\\displaystyle {item[-1]}$"
                                col_latex.code(item[-1], language="latex")
                                col_render.markdown(render_latex)
                                img = np.asarray(item[0])
                                col_org.image(img)
                    else:
                        for idx, (name, org, latex) in enumerate(
                            zip(display_name, origin_img, latex_code)
                        ):
                            container = streamlit.container()
                            col_latex, col_render, col_org = container.columns(
                                3, gap="large"
                            )

                            if idx == 0:
                                col_latex.header("Predicted LaTeX")
                                col_render.header("Rendered Image")
                                col_org.header("Cropped Image")

                            render_latex = f"$\\displaystyle {latex}$"
                            col_latex.code(latex, language="latex")
                            col_render.markdown(render_latex)
                            org = np.asarray(org)
                            col_org.image(org)

                except Exception as e:
                    streamlit.error(e)
        else:
            streamlit.error("Please upload an image.")


if __name__ == "__main__":
    # print(f"Is CUDA available: {torch.cuda.is_available()}")
    # # True
    # print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    # Tesla T4
    api()
