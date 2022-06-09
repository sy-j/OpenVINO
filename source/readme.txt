1. save tensorflow model(.pb file)
python infoboss_save_model.py

2. openvino env initialize
source /opt/intel/openvino_2021/bin/setupvars.sh

3. Make IR file(model optimization)
python /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py  --input_model /home/systartup/data/infoboss_frozen_model.pb  --input x,x_1  --input_shape '[2500,3000,4], [2500,64]'  --model_name 'infoboss_frozen_model_2500'

	mo.py
	input_model = 1에서 저장한 pb파일
	input = 1에서 저장한 모델에서 input node 이름 (뜯어서 확인해야함)
	input_shape = '[개수(batchsize), 길이, 너비]'
	model_name = IR파일 이름
		batchsize = 1 ~ 5000

4. inference
python infoboss_openvino_inference.py  -m /home/systartup/anaconda3/envs/openvino/sources/infoboss_frozen_model_2500.xml


모든 절차에서 경로 확인 필수