# XycarState.msg

# 컨트롤러가 추종해야 할 목표 지점과 각도 정보
# x: BEV 이미지 상의 목표 x좌표 (pixel)
# y: BEV 이미지 상의 기준 y좌표 (pixel)
# z: 현재 차선의 각도 (radian)
geometry_msgs/Point target_point

# 현재 차량의 주행 모드 ("center", "left", "right")
string drive_mode
