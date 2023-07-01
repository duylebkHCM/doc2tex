from collections import OrderedDict


class CFG:
    image_name = [
        "de8a312222.png",
        "4d73f273d6.png",
        "44d0df668c.png",
        "68e78c6272.png",
    ]

    img_dir = (
        "../../analysis/dicta_analysis/dataset/data_20211115/formula_images_processed"
    )
    csv_dir = "../../analysis/dicta_analysis/best_result/report_papers_20220621-004431-LR0.0005-batchsize32_best_accuracy.pth/cpu_batch1_beam5_1806_formula_images_processed_img_metric.csv"

    output_dir = (
        "../../analysis/dicta_analysis/decoder_attn/new_2105_run/match_imgs_with_labels"
    )

    class Decoder:
        condition = r"(len < 50 & len > 30)%iscorrect: True"
        display_last = False
        save_seperate_fig = True
        num_samples = -1
        smooth = False
        put_text = True
        down_sample = 1 / 1.5

    class FE:
        target_layers = "proj"
        output_dir = "../analysis/feature_map_resnet/short"
        downsample = 1
        condition = r"(len < 80 & len > 60)%iscorrect: True"

    class Encoder:
        head_fusion = "max"
        discard_ratio = 0.9
        output_dir = "../analysis/vit_visualize/match_imgs/long/"
        downsample = 1 / 1.5
        condition = r"(len < 100 & len > 80)%iscorrect: True"
        debug = True

    @staticmethod
    def get_attr(obj_):
        return_dict = OrderedDict()
        for k, v in obj_.items():
            if k.startswith("__"):
                continue
            return_dict[k] = v
        return return_dict
