-->원하는 비드 색 따서 라벨링
python3 convert_masks.py --input ~/replicator_output/oblique_v5_batch_51373/ --bead_color "93,220,11"


-->prac1/train_unet_*.py 학습 후 test inference.py 실행하는법

test_inference.py의 파라미터를 train_unet_*.py파라미터와 일치시킨다
모델 저장 경로 또한 일치시킨 후
python3 test_inference.py --checkpoint  checkpoints_3000/april_resumed6.pth  --input /home/kim/Downloads/beadlearn/fitimage/001.jpg --threshold 0.1

터미널 실행.

eval_real.py도 똑같이 일치시킨 다음, 그 파일만 실행시키면 설정 폴더 내 이미지에 대한 dice score 출력 가능


side_photo_DR.py는 isaac sim 내에서 활용할 수 있는 카메라 위치 변환 스크립트이다.

DR실행 시 카메라 위치를 변화시키며 물체를 촬영 가능하다.

합성 데이터 통합 코드를 통해 DR스크립트를 이용해 추출한 이미지들을 convert_masks.py로 binary mask 추출 후 하나의 폴더에 통합한다.
