# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#
# test
#
find_package(GTest 1.9.0 REQUIRED)
find_package(Threads REQUIRED)    # required for gtest
find_package(blaze 3.5 REQUIRED)

# ~~~~~~~~~~ Our tests ~~~~~~~~~~~~~~
add_executable(test-blazefeo
    math/simd/RegisterMatrixTest.cpp
    math/simd/DynamicRegisterMatrixTest.cpp
    math/simd/HsumTest.cpp

    math/dense/StaticMatrixPointerTest.cpp
    math/dense/DynamicMatrixPointerTest.cpp
    math/dense/GemmTest.cpp
    math/dense/SyrkTest.cpp
    math/dense/PotrfTest.cpp
    math/dense/TrmmTest.cpp
    math/dense/TrsvTest.cpp

    math/panel/StaticPanelMatrixTest.cpp
    math/panel/DynamicPanelMatrixTest.cpp
    math/panel/GemmTest.cpp
    math/panel/PotrfTest.cpp

    math/expressions/PMatTransExprTest.cpp
    math/expressions/AssignPanelDenseTest.cpp
    math/expressions/AssignDensePanelTest.cpp

    math/views/RowTest.cpp
    math/views/SubmatrixTest.cpp
)

target_link_libraries(test-blazefeo PRIVATE
    blazefeo
    blaze::blaze
    GTest::GTest
    GTest::Main
    gmock
    openblas
    ${CMAKE_THREAD_LIBS_INIT}    # Needed for gtest
)

target_compile_definitions(test-blazefeo 
    PRIVATE "BLAZE_USER_ASSERTION=1;BLAZE_INTERNAL_ASSERTION=1"
)

gtest_discover_tests(test-blazefeo)

if (BLAZEFEO_WITH_BLASFEO)
    add_subdirectory("blasfeo")
endif ()