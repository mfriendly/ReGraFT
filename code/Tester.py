from datetime import datetime

import matplotlib
import numpy as np

from data_utils import *
from training_utils import *

matplotlib.use("Agg")
import os
import warnings

warnings.filterwarnings("ignore")

def Test(
    config,
    args,
    model_,
    test_loader,
    adjs,
    scaler_name,
    standard_scaler_stats_dict,
    diff,
    initial_values_date_dict,
    mean_wk="6wk",
):

    model_.eval()
    nation = args.nation
    save_path = args.save_path
    INPUTID = args.INPUTID
    SEED = args.SEED
    MODEL = args.MODEL
    past_steps = args.past_steps

    (list_of_states, list_of_abbr, states2abbr,
     adjs) = retrieve_metadata_of_nation(config, args.nation, args.input_path,
                                         args.device)

    week_metrics_list, preds_dict = create_result_dictionaries(
        nation, list_of_abbr)
    result_path = f"../results/covid/{nation}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    csv_file_name = f"../results/covid/{nation}/{MODEL}_{INPUTID}_metric__{SEED}.csv"
    if os.path.exists(csv_file_name):
        os.remove(csv_file_name)
    for index, datat in enumerate(test_loader):

        (
            raw_predictions,
            attn_output_weights,
            encoder_sparse_weights,
            decoder_sparse_weights,
            encoder_adjs_output_list,
            decoder_adjs_output_list,
        ) = model_(datat, adjs, global_step=100000)

        if encoder_sparse_weights != None:
            name = "encoder_sparse"
            sparse_matrix_save_npy(args, encoder_sparse_weights.squeeze(),
                                   index, name)
            name = "decoder_sparse"
            sparse_matrix_save_npy(args, decoder_sparse_weights.squeeze(),
                                   index, name)
        datetime.now().strftime("%Y%m%d_%H%M%S")

        for ii, mat in enumerate(encoder_adjs_output_list):
            if ii % 7 == 0 or ii in [6, 20, 41]:
                name = "encoder"
                adap_matrix_save_npy_png(args, mat, index, ii, name)

        for ii, mat in enumerate(decoder_adjs_output_list):
            if ii % 7 == 0:
                name = "decoder"
                adap_matrix_save_npy_png(args, mat, index, ii, name)

        start_date_, end_date_ = retrieve_start_end_dates_of_batch(datat)
        if raw_predictions.dim() == 3:
            raw_predictions = raw_predictions.unsqueeze(-1)
        for node_id, abbr in enumerate(list_of_abbr):

            predictions_abbr_ = raw_predictions[:, node_id, :, :]
            if abbr not in preds_dict:
                preds_dict[abbr] = {}
            predictions_abbr_to_scale = predictions_abbr_.cpu().detach().numpy(
            ).reshape(-1, 1)

            if args.scaler_name == "":
                inverse_transformed_predictions = inverse_transform_predictions(
                    config, args, predictions_abbr_to_scale, abbr,
                    standard_scaler_stats_dict)

            if args.diff:
                inverse_transformed_predictions = reconstruct_series_to_array(
                    inverse_transformed_predictions,
                    initial_values_date_dict[abbr],
                    start_date_,
                    end_date_,
                    "%Y-%m-%d",
                    abbr,
                )
            inverse_transformed_predictions = replace_negatives_and_interpolate(
                inverse_transformed_predictions)

            csv_path = f"../data/x_data_unscaled/{nation}/{str(node_id).zfill(2)}_{abbr}_label_unscaled.csv"

            combined_tar = get_values_in_date_range(csv_path, start_date_,
                                                    end_date_, "new_confirmed")
            actuals = combined_tar[past_steps:]
            combined_tar_np = np.array(combined_tar).reshape(-1, 1)

            dates_ = retrieve_pandas_date_list(start_date_, end_date_)
            preds_dict = update_predictions(
                preds_dict=preds_dict,
                abbr=abbr,
                start_date=start_date_,
                end_date=end_date_,
                inverse_transformed_predictions=inverse_transformed_predictions,
                combined_tar=combined_tar_np,
                past_steps=past_steps,
                dates=dates_,
                save_path=save_path,
            )

            labels_cpu = datat["outputs"].detach().cpu()
            process_and_save_metrics(
                config,
                args,
                index,
                actuals,
                inverse_transformed_predictions,
                predictions_abbr_to_scale,
                labels_cpu,
                node_id,
                abbr,
                week_metrics_list,
                save_path,
                MODEL,
                csv_file_name,
                mean_wk,
            )

    apply_mean_calculation(config, args, csv_file_name)
    return config, preds_dict
