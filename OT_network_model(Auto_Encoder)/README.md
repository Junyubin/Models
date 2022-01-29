# ot network model
# 평균 성능 accuray 약 95.5%

    KISA 검증       

    1. 실행 방법 (kisa_ot_detection.py)
        1.1	Terminal에서 해당 py 파일이 들어있는 디렉토리로 이동
            1)	bash
            2)	[ex] cd home/ctilab/git_clone/dti.models/ot_network_model

        2.1 파이썬 환경에서 해당 py 파일 실행
            1) python3.7 kisa_ot_detection.py       
            
        3.1 모드 설정
            1) 라이브러리 로드 후 모드 입력 실행
                : Enter the mode 표시 확인
                : 학습일 경우 train 입력
                : 예측일 경우 predict 입력
    
    2. 학습 실행 (mode = train)
        2.1 버전 세팅
            1) '연월일_시' 형태로 지정
            2) [ex] 20211215_15
            
        2.2 config  폴더에 저장되어 있는 json 형식 파일 로드
            1) 필요시 config 폴더에 json 파일 수정 필요
            
        2.3 데이터 전처리
            1) Minmaxscaling 진행
            2) scaler fitting 후 pickle 형식으로 자동 저장 (저장 경로 - ot_model/버전)
            
        2.4 모델학습 (Fitting & Save Model)
            1) DNN Autoencoder 실행
            2) 모델 Fitting 후 h5 형식으로 자동 저장 (저장 경로 - otmodel/버전)
            
    3. 예측 실행 (mode = predict)
        3.1 버전 세팅
            1) 학습 된 버전 중 가장 최신 버전으로 자동 설정
            2) '연월일_시' 형태로 지정
            
        3.2 config  폴더에 저장되어 있는 json 형식 파일 로드
            1) 필요시 config 폴더에 json 파일 수정 필요
            
        3.3 데이터 전처리
            1) 지정 된 버전에서 학습 된 scaler 호출
            2) scaler load 후 transform 진행
            
        3.4 모델예측
            1) 지정 된 버전에서 학습 된 DNN 모델 호출
            2) 데이터 예측 후 검증 시행
            3) 검증은 Accuracy score와 Confusion Matrix 점수로 확인
            3) 검증 완료 후 데이터 Insert (DB)
            