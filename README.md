# DEMO apps for SURVEILLANCE project


## How to build & run the app?

> Make sure you're standing at repo's root @ `.../demo`

- Step 1: Get `.env` file & put it in repo's root, then source it: `source .env`

- Step 2: Build app: `make build`

- Step 3: Run app: `make up` & Down app: `make down`

- To run:
`docker run -it --rm --runtime=nvidia --gpus all manhckv/test:latest`

- To run infer:
`python3 src/src_infer/infer_yolo_light.py`

- To zip docker:
`docker save -o export_image.tar ${IMAGE_NAME}:${IMAGE_VERSION}`

- Docker copy
`docker cp 2399eae3124d:/src/predict.txt . `