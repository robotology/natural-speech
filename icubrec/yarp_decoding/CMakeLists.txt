# Copyright (C) 2017 iCub Facility - Istituto Italiano di Tecnologia
# Author: Bertrand Higy
# CopyPolicy: Released under the terms of the GNU GPL v3.0.
set(PROJECTNAME natural-speech)
set(APPNAME icubrec)
set(PGMNAME rctrld_yarphear_asr)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${YARP_INCLUDE_DIRS})
include_directories(${ACE_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_executable(${PGMNAME} rctrld_yarphear_asr.cpp)

target_link_libraries(${PGMNAME} ${YARP_LIBRARIES})
target_link_libraries(${PGMNAME} ${ACE_LIBRARIES})

install(TARGETS ${PGMNAME} DESTINATION bin)

# Context and application files
file(GLOB lua ${CMAKE_CURRENT_SOURCE_DIR}/*.lua)
file(GLOB xml ${CMAKE_CURRENT_SOURCE_DIR}/*.xml)
file(GLOB py ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
file(GLOB sh ${CMAKE_CURRENT_SOURCE_DIR}/*.sh)
yarp_install(FILES ${lua} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${PROJECTNAME})
yarp_install(FILES ${xml} DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})
yarp_install(PROGRAMS ${py} DESTINATION bin)
yarp_install(PROGRAMS ${sh} DESTINATION bin)
