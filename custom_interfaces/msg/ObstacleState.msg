# ObstacleState.msg

float32 distance_m    # 라이다로 측정한 장애물까지의 평균 거리 (미터 단위)
string position     # 차선을 기준으로 한 장애물의 상대 위치 ("left", "right", "none", "unknown")

uint64 vehicle_count # 장애물 차량의 수