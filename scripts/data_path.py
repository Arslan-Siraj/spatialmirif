from pathlib import Path

###########
# 0.zarr data path
# S1_N2
data_path_zarr_S1_N2 = Path(r"D:\Project_Cemos\with_pymir\Slide_1\S1_N2\0.zarr")
data_path_zarr_S1_N2_info = Path(r"D:\Project_Cemos\with_pymir\Slide_1\S1_N2\MMI2_AD_Slide_1_090126_TissuePlus_Transmission_no_SF_INFO.txt")

data_path_if_tif_S1_N2_raw = Path(r"D:\Project_Cemos\with_pymir\Slide_1\S1_N2\S1_N2_IF_raw.tif")

data_path_if_nrrd_S1_N2 = Path(r"D:\Project_Cemos\with_pymir\Slide_1\S1_N2\S1_N2_IF.nrrd")
data_path_if_nrrd_S1_N2_mask = Path(r"D:\Project_Cemos\with_pymir\Slide_1\S1_N2\S1_N2_IF_mask.nrrd")

data_path_full_slide_nrrd_S1_N2 = Path(r"D:\Project_Cemos\test_data_for_pipeline\PPS_Slide_1_BF_10x.nrrd")

output_path_S1_N2 = Path(r"D:\Project_Cemos\with_pymir\Slide_1\S1_N2\output_with_spatialmirif")

data_path_mir_aligned_with_if = Path(r"D:\Project_Cemos\with_pymir\Slide_1\S1_N2\S1_N2_MIR_1650_pymir_align.nrrd")

data_path_if_aligned = Path(r"D:\Project_Cemos\with_pymir\Slide_1\S1_N2\S1_N2_IF_align.nrrd")
data_path_if_mask_aligned = Path(r"D:\Project_Cemos\with_pymir\Slide_1\S1_N2\S1_N2_IF_mask_pymir_align.nrrd")