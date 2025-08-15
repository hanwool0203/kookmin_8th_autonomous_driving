cmake_minimum_required(VERSION 3.8)
project(mission_cone_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 필요한 패키지를 찾습니다.
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 2. msg 파일들을 지정하여 인터페이스 코드를 생성합니다.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ClusterData.msg"
  "msg/ConeData.msg"
  # 우리의 메시지가 geometry_msgs에 의존함을 명시합니다.
  DEPENDENCIES geometry_msgs
)

# 3. 설치 및 테스트 관련 설정입니다.
ament_export_dependencies(rosidl_default_runtime)
ament_package()

