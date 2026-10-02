# Ubuntu's 3.5 package exports nonexistent static archives. Expose the actual
# unmodified distribution shared libraries instead; no system files are edited.
get_filename_component(_isaacmin_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(_osd_prefix "${_isaacmin_root}/.tools/native/usr")
set(OPENSUBDIV_INCLUDE_DIR "${_osd_prefix}/include" CACHE PATH "Private OpenSubdiv headers" FORCE)
set(OPENSUBDIV_INCLUDE_DIRS "${_osd_prefix}/include")
foreach(_kind CPU GPU)
  if(NOT TARGET OpenSubdiv::osd${_kind})
    add_library(OpenSubdiv::osd${_kind} SHARED IMPORTED)
    set_target_properties(OpenSubdiv::osd${_kind} PROPERTIES
      IMPORTED_LOCATION "${_osd_prefix}/lib/aarch64-linux-gnu/libosd${_kind}.so.3.5.0"
      INTERFACE_INCLUDE_DIRECTORIES "${_osd_prefix}/include")
  endif()
endforeach()
set(OpenSubdiv_FOUND TRUE)
