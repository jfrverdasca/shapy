FROM python:3.10-bullseye

RUN apt-get update && \
    apt-get install -y libgl1 libegl1 libegl-mesa0

RUN pip3 install torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install opencv-python chumpy imageio jpeg4py joblib kornia loguru matplotlib numpy open3d Pillow PyOpenGL pyrender scikit-image scikit-learn scipy smplx tqdm trimesh yacs fvcore nflows PyYAML && \
    pip3 install omegaconf==2.0.6


ENTRYPOINT ["python3"]