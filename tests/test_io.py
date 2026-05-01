import pytest
import os
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from core.io_utils import load_multiframe_dicom, create_output_directories

def test_create_output_directories(tmp_path):
    # Change current working directory to tmp_path for this test
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        paths = create_output_directories("test_case")
        assert "png_ap" in paths
        assert os.path.exists(paths["png_ap"])
        assert "pt_lat" in paths
        assert os.path.exists(paths["pt_lat"])
    finally:
        os.chdir(original_cwd)

def test_load_multiframe_dicom(tmp_path):
    # Create a dummy multiframe dicom
    filename = str(tmp_path / "dummy.dcm")

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7' # Secondary Capture
    file_meta.MediaStorageSOPInstanceUID = '1.2.3'
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.NumberOfFrames = 10
    ds.RecommendedDisplayFrameRate = 15.0
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(filename)

    loaded_ds, fps = load_multiframe_dicom(filename)
    assert loaded_ds.NumberOfFrames == 10
    assert fps == 15.0

def test_load_multiframe_dicom_invalid(tmp_path):
    # File doesn't exist
    with pytest.raises(FileNotFoundError):
        load_multiframe_dicom("nonexistent.dcm")

    # Not multiframe
    filename = str(tmp_path / "single.dcm")
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.NumberOfFrames = 1
    ds.save_as(filename)

    with pytest.raises(ValueError):
        load_multiframe_dicom(filename)
