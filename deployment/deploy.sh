x-common-variables: &common-variables
  IMAGE_NAME: ${IMAGE_NAME}
  IMAGE_VERSION: ${IMAGE_VERSION}
  CONTAINER_NAME: ${CONTAINER_NAME}
  PORT: ${PORT}

  CAM1_URL: ${CAM1_URL}
  CAM2_URL: ${CAM2_URL}
  CAM3_URL: ${CAM3_URL}

  DATA_ROOT_PATH: ${DATA_ROOT_PATH}
  CAM1_VIDEO_PATH: ${CAM1_VIDEO_PATH}
  CAM2_VIDEO_PATH: ${CAM2_VIDEO_PATH}
  CAM3_VIDEO_PATH: ${CAM3_VIDEO_PATH}

  HEAD_MODEL_PATH: ${HEAD_MODEL_PATH}
  ULD_MODEL_PATH: ${ULD_MODEL_PATH}
  HUMAN_MODEL_PATH: ${HUMAN_MODEL_PATH}
  ULD_DOOR_MODEL_PATH: ${ULD_DOOR_MODEL_PATH}

  DEVICE: ${DEVICE}
  LOGIN_LOG_PATH: ${LOGIN_LOG_PATH}
  UPLOAD_FOLDER: ${UPLOAD_FOLDER}

x-volumes: &default-volume
  volumes:
    - ${DATA_ROOT_PATH}:${DATA_ROOT_PATH}

x-deploy: &default-deploy
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: [ "0" ]
            capabilities: [ gpu ]

services:
  demo-app:
    image: "${IMAGE_NAME}:${IMAGE_VERSION}"
    container_name: ${CONTAINER_NAME}
    environment: *common-variables
    restart: always
    network_mode: "host"
    <<: [ *default-volume, *default-deploy ]
    command: >
      streamlit run src/app.py 
      --server.address "0.0.0.0" 
      --server.port "${PORT}"

