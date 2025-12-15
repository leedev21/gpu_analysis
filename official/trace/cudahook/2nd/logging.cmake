include(FetchContent)
FetchContent_Declare(logging
    GIT_REPOSITORY git@git.git
    GIT_TAG 4.0.34
)
FetchContent_MakeAvailable(logging)
include_directories(${logging_dir})